"""
Vectorizer Module

This module handles the vectorization of data using LanceDB and a local Ollama instance.
"""

import lancedb
import pyarrow as pa  # Add this import for schema types
from typing import List, Dict
import requests
from tqdm import tqdm
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants
LANCEDB_PATH = "./lancedb"  # Lokaler Pfad zur LanceDB-Datenbank
OLLAMA_URL = "http://localhost:11434/api/embeddings"
# Modell kann jetzt per Umgebungsvariable gewählt werden: mxbai-embed-large:latest oder phi4:latest
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mxbai-embed-large:latest")
FILE_EXTENSION_FILTERS = [".al", ".json"]  # Erlaubte Dateiendungen

# Mehrere Root-Dirs als Liste
ROOT_DIRS = [
    r"/home/kosta/Repos/DevOps/Product_KBA/Product_KBA_BC_AL/app/",
    r"/home/kosta/Repos/DevOps/Product_MED/Product_MED_AL/app/",
    r"/home/kosta/Repos/DevOps/Product_MED_Tech365/Product_MED_Tech/app/",
    r"/home/kosta/Repos/GitHub/StefanMaron/MSDyn365BC.Code.History/BaseApp/Source/Base Application"
    
]

# Initialize LanceDB
def initialize_lancedb() -> lancedb.db.DBConnection:
    return lancedb.connect(LANCEDB_PATH)

def compute_content_hash(content: str) -> str:
    """
    Compute a SHA256 hash for the given content.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_object_info(content: str, filename: str) -> dict:
    """
    Extrahiere Objekt-Id, Objektart, Objekt Name, Namespace aus dem Content oder Dateinamen.
    Diese Funktion ist ein Platzhalter und sollte je nach Dateiformat angepasst werden.
    """
    import re
    object_id = ""
    object_type = ""
    object_name = ""
    namespace = ""
    # Versuche, Objektinformationen aus dem Inhalt zu extrahieren
    match = re.search(r'^(table|page|codeunit|report|enum|interface|query|xmlport|controladdin|profile|permissionset|entitlement|enumextension|tableextension|pageextension|reportextension|permissionsetextension|dotnet|label)\s+(\d+)\s+("[^"]+"|\w+)', content, re.MULTILINE | re.IGNORECASE)
    if match:
        object_type = match.group(1)
        object_id = match.group(2)
        object_name = match.group(3).strip('"')
    # Namespace ggf. aus dem Inhalt extrahieren (z.B. "namespace = 'MyNamespace';")
    ns_match = re.search(r'namespace\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
    if ns_match:
        namespace = ns_match.group(1)
    # Spezialfall: KVSKBA-Objekte -> Namespace aus Verzeichnisname
    if object_name.startswith("KVSKBA"):
        # Versuche, das übergeordnete Verzeichnis aus dem filename zu extrahieren
        # filename ist nur der Dateiname, daher muss der Aufrufer das Verzeichnis mitgeben
        # Wir erwarten, dass extract_object_info im Kontext von collect_files aufgerufen wird,
        # daher kann dort das Verzeichnis übergeben werden.
        pass  # Wird unten in collect_files behandelt
    return {
        "object_id": object_id,
        "object_type": object_type,
        "object_name": object_name,
        "namespace": namespace
    }

# Vectorize Data
def vectorize_data(data: List[Dict[str, str]]) -> None:
    """
    Vectorize the provided data and store it in LanceDB.

    Args:
        data (List[Dict[str, str]]): List of objects to vectorize.
            Jeder Eintrag sollte zusätzlich 'filename' und 'directory' enthalten.
    """
    db = initialize_lancedb()
    
    # Use pyarrow.Schema instead of dict
    table_schema = pa.schema([
        ("id", pa.string()),
        ("content", pa.string()),
        ("embedding", pa.list_(pa.float32(), 1024)),  # mxbai-embed-large: 1024-dim
        ("filename", pa.string()),
        ("directory", pa.string()),
        ("content_hash", pa.string()),
        # Neue Felder:
        ("object_id", pa.string()),
        ("object_type", pa.string()),
        ("object_name", pa.string()),
        ("namespace", pa.string()),
    ])
    
    try:
        # Try to open existing table
        table = db.open_table("namespace_vectors")
        # Prüfe, ob neue Spalten fehlen und gib ggf. einen Hinweis aus
        existing_fields = set(table.schema.names)
        missing_fields = [field for field in ["object_id", "object_type", "object_name", "namespace"] if field not in existing_fields]
        if missing_fields:
            print(f"FEHLER: Die Tabelle existiert bereits, aber folgende Felder fehlen: {missing_fields}.")
            print("Bitte lösche die Tabelle 'namespace_vectors' in LanceDB und lasse das Skript erneut laufen, damit das Schema korrekt angelegt wird.")
            print("Alternativ: Migriere die Tabelle manuell mit den neuen Feldern.")
            return  # Abbruch, um weitere Fehler zu vermeiden
    except Exception:
        # Create new table if it doesn't exist
        table = db.create_table("namespace_vectors", schema=table_schema)

    # Lade alle existierenden (filename, content_hash) Paare EINMALIG
    existing_pairs = set()
    try:
        df = table.to_pandas()
        existing_pairs = set(zip(df["filename"], df["content_hash"]))
    except Exception:
        pass  # Tabelle ist evtl. noch leer

    start_time = time.time()
    batch = []
    batch_size = 64  # Passe die Batchgröße ggf. an
    max_workers = 10  # Passe die Thread-Anzahl ggf. an

    def embedding_task(item):
        content_hash = compute_content_hash(item["content"])
        pair = (item.get("filename", ""), content_hash)
        if pair in existing_pairs:
            return None  # Already exists, skip
        filename = item.get("filename", "")
        has_other_hash = any(f == filename and h != content_hash for (f, h) in existing_pairs)
        # Extrahiere Objektinfos
        obj_info = extract_object_info(item["content"], item["filename"])
        return {
            "item": item,
            "content_hash": content_hash,
            "pair": pair,
            "filename": filename,
            "has_other_hash": has_other_hash,
            "obj_info": obj_info,
        }

    with tqdm(total=len(data), desc="Vektorisieren", unit="Dokument") as pbar:
        # Schritt 1: Vorfiltern, was wirklich verarbeitet werden muss
        filtered = []
        skipped = 0
        for item in data:
            res = embedding_task(item)
            if res:
                filtered.append(res)
            else:
                skipped += 1
                pbar.update(1)
        total_to_process = len(filtered)
        print(f"{skipped} Dateien übersprungen (bereits vorhanden), {total_to_process} zu vektorisieren.")

        # Schritt 2: Embeddings parallel generieren
        def embedding_worker(res):
            print(f"Generiere Embedding für: {res['item']['filename']}")  # Logging
            embedding = generate_embedding(res["item"]["content"])
            return {**res, "embedding": embedding}

        results = []
        processed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(embedding_worker, res) for res in filtered]
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                    # Delete nur wenn nötig
                    if res["has_other_hash"]:
                        del_filter_str = f"filename = '{res['filename']}'"
                        table.delete(where=del_filter_str)
                    batch.append({
                        "id": res["item"]["id"],
                        "content": res["item"]["content"],
                        "embedding": res["embedding"],
                        "filename": res["filename"],
                        "directory": res["item"].get("directory", ""),
                        "content_hash": res["content_hash"],
                        # Neue Felder:
                        "object_id": res["obj_info"].get("object_id", ""),
                        "object_type": res["obj_info"].get("object_type", ""),
                        "object_name": res["obj_info"].get("object_name", ""),
                        "namespace": res["obj_info"].get("namespace", ""),
                    })
                    existing_pairs.add(res["pair"])
                    processed += 1
                    # Batch-Insert
                    if len(batch) >= batch_size:
                        table.add(batch)
                        batch.clear()
                except Exception as e:
                    print(f"Fehler bei Verarbeitung: {e}")
                # Fortschritt nur alle 10 Schritte aktualisieren
                if idx % 10 == 0 or idx == total_to_process:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (total_to_process - idx)
                    pbar.set_postfix_str(f"Ø {avg_time:.2f}s, Rest {remaining/60:.1f}min")
                pbar.update(1)
        # Restliche Einträge einfügen
        if batch:
            table.add(batch)
            batch.clear()
        print(f"{processed} Dateien vektorisiert und gespeichert.")

def generate_embedding(content: str) -> List[float]:
    """
    Generate an embedding for the given content using the local Ollama instance.

    Args:
        content (str): The content to embed.

    Returns:
        List[float]: The embedding vector.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": content
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)  # Timeout erhöht
        response.raise_for_status()
        result = response.json()
        # Ollama gibt das Embedding unter "embedding" zurück
        embedding = result.get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise ValueError("No embedding returned from Ollama.")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * 1024  # Fallback auf Nullvektor

def collect_files(root_dir: str, extensions: list) -> List[Dict[str, str]]:
    """
    Sammelt rekursiv alle Dateien mit den angegebenen Extensions ab root_dir.
    Gibt eine Liste von Dicts mit id, content, filename, directory zurück.
    """
    result = []
    file_count = 0
    dir_count = 0
    
    print(f"Sammle Dateien mit Endungen {', '.join(extensions)}...")
    
    total_dirs = sum([len(dirpath) for dirpath, _, _ in os.walk(root_dir)])
    
    for dirpath, _, filenames in os.walk(root_dir):
        dir_count += 1
        if dir_count % 10 == 0:
            print(f"Verarbeite Verzeichnis {dir_count}/{total_dirs}: {os.path.relpath(dirpath, root_dir)}")
        
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in extensions):
                file_count += 1
                full_path = os.path.join(dirpath, fname)
                try:
                    with open(full_path, encoding="utf-8") as f:
                        content = f.read()
                    obj_info = extract_object_info(content, fname)
                    # Spezialfall: KVSKBA-Objekte -> Namespace aus übergeordnetem Verzeichnis
                    if obj_info.get("object_name", "").startswith("KVSKBA"):
                        # Hole das übergeordnete Verzeichnis (direkt über dem Dateinamen)
                        rel_dir = os.path.relpath(dirpath, root_dir)
                        # Namespace ist der letzte Teil des relativen Verzeichnispfads
                        ns_candidate = os.path.basename(rel_dir)
                        # Setze Namespace, falls nicht schon im Content gefunden
                        if not obj_info.get("namespace"):
                            obj_info["namespace"] = ns_candidate
                    result.append({
                        "id": os.path.relpath(full_path, root_dir),
                        "content": content,
                        "filename": fname,
                        "directory": os.path.relpath(dirpath, root_dir),
                        "object_id": obj_info.get("object_id", ""),
                        "object_type": obj_info.get("object_type", ""),
                        "object_name": obj_info.get("object_name", ""),
                        "namespace": obj_info.get("namespace", ""),
                    })
                    
                    if file_count % 50 == 0:
                        print(f"Dateien gefunden: {file_count} (aktuell: {fname})")
                        
                except Exception as e:
                    print(f"Fehler beim Lesen von {full_path}: {e}")
    
    print(f"Dateisammlung abgeschlossen. Insgesamt {file_count} Dateien gefunden in {dir_count} Verzeichnissen.")
    return result

def main():
    all_data = []
    print("Starte Vektorisierung für folgende Verzeichnisse:")
    for root_dir in ROOT_DIRS:
        print(f"  - {root_dir} (nur {', '.join(FILE_EXTENSION_FILTERS)})")
        data = collect_files(root_dir, FILE_EXTENSION_FILTERS)
        print(f"{len(data)} Dateien gefunden in {root_dir}.")
        all_data.extend(data)
    print(f"Insgesamt {len(all_data)} Dateien gefunden. Starte Vektorisierung...")
    vectorize_data(all_data)
    print("Vektorisierung abgeschlossen.")

if __name__ == "__main__":
    main()

# Beispiel-Skript zum Löschen der Tabelle 'namespace_vectors' in LanceDB

# import lancedb

# LANCEDB_PATH = "./lancedb"

# def drop_namespace_vectors():
#     db = lancedb.connect(LANCEDB_PATH)
#     db.drop_table("namespace_vectors")
#     print("Tabelle 'namespace_vectors' wurde gelöscht.")

# if __name__ == "__main__":
#     drop_namespace_vectors()