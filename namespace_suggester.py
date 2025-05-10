import os
import re
import csv
import requests
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import lancedb
import numpy as np

HC_ROOT = "/home/kosta/Repos/DevOps/Product_MED/Product_MED_AL/app/"    # Passe ggf. an
MTC_ROOT = "/home/kosta/Repos/DevOps/Product_MED_Tech365/Product_MED_Tech/app/"  # Passe ggf. an
CSV_OUTPUT = "namespace_suggestions.csv"

OBJECT_PATTERN = re.compile(r'^(table|page|codeunit|report|xmlport|query|enum|interface|controladdin|pageextension|tableextension|enumextension|profile|dotnet|entitlement|permissionset|permissionsetextension|reportextension|enumvalue|entitlementset|entitlementsetextension)\s+(\d+)?\s*"?([\w\d_]+)"?', re.IGNORECASE)

PREFIX_HC = "KVSMED"
PREFIX_MTC = "KVSMTC"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "phi4:latest"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")
LANCEDB_PATH = "./lancedb"
LANCEDB_TABLE = "namespace_vectors"

ALLOWED_NAMESPACES = {
    # Microsoft Standard
    "Warehouse", "Utilities", "System", "Service", "Sales", "RoleCenters", "Purchases", "Projects", "Profile",
    "Pricing", "OtherCapabilities", "Manufacturing", "Invoicing", "Inventory", "Integration", "HumanResources",
    "Foundation", "FixedAssets", "Finance", "eServices", "CRM", "CostAccounting", "CashFlow", "Bank", "Assembly", "API",
    # KUMAVISION Base (KBA)
    "UDI", "Call", "LIF", "OrderQuote", "InventorySummary", "Common",
    # HC/MTC-spezifisch
    "EDocuments", "ECE", "MDR"
}

def find_al_files(root: str) -> List[str]:
    al_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".al"):
                al_files.append(os.path.join(dirpath, f))
    return al_files

def extract_object_info(filepath: str) -> Optional[Tuple[str, str]]:
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                match = OBJECT_PATTERN.match(line.strip())
                if match:
                    obj_type = match.group(1)
                    obj_name = match.group(3)
                    return obj_type, obj_name
    except Exception:
        pass
    return None

def remove_prefix(name: str, prefix: str) -> str:
    # Korrigiert: Case-insensitive Prefix-Entfernung
    if name.upper().startswith(prefix.upper()):
        return name[len(prefix):]
    return name

def get_embedding(text: str) -> List[float]:
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        embedding = result.get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise ValueError("No embedding returned from Ollama.")
        return embedding
    except Exception as e:
        print(f"Embedding-Fehler: {e}")
        return [0.0] * 1024

def retrieve_context(object_type: str, object_name: str, top_k: int = 3) -> List[Dict]:
    # Initialisiere LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(LANCEDB_TABLE)
    # Erzeuge Embedding für Suchtext
    query_text = f"{object_type} {object_name}"
    query_emb = get_embedding(query_text)
    # Suche nach ähnlichsten Einträgen
    try:
        results = table.search(query_emb).limit(top_k).to_list()
        return results
    except Exception as e:
        print(f"LanceDB Retrieval-Fehler: {e}")
        return []

NAMESPACE_OVERVIEW = """
1. Microsoft Standard:
   - Warehouse: Lagerverwaltung, Lagerbewegungen, Lagerorte, Lagerlogistik
   - Utilities: Hilfsfunktionen, technische Werkzeuge, allgemeine Tools
   - System: Systemobjekte, technische Infrastruktur, Systemprozesse
   - Service: Serviceaufträge, Serviceartikel, Serviceprozesse
   - Sales: Verkauf, Verkaufsbelege, Verkaufsprozesse
   - RoleCenters: Rollencenter, Benutzeroberflächen für Rollen
   - Purchases: Einkauf, Einkaufsbelege, Einkaufsprozesse
   - Projects: Projektmanagement, Projektaufträge, Projektplanung
   - Profile: Benutzerprofile, Einstellungen für Benutzer
   - Pricing: Preisfindung, Preislisten, Rabatte
   - OtherCapabilities: Sonstige Fähigkeiten, Zusatzfunktionen
   - Manufacturing: Fertigung, Produktionsaufträge, Stücklisten
   - Invoicing: Rechnungsstellung, Fakturierung
   - Inventory: Bestandsführung, Lagerbestände
   - Integration: Schnittstellen, externe Anbindungen, Integration
   - HumanResources: Personalverwaltung, Mitarbeiterdaten
   - Foundation: Grundfunktionen, Basiskomponenten
   - FixedAssets: Anlagenbuchhaltung, Anlagegüter
   - Finance: Finanzbuchhaltung, Buchungen, Konten
   - eServices: Elektronische Dienste, Online-Services
   - CRM: Kundenbeziehungsmanagement, Kontakte, Aktivitäten
   - CostAccounting: Kostenrechnung, Kostenstellen
   - CashFlow: Liquiditätsplanung, Zahlungsströme
   - Bank: Bankkonten, Zahlungsverkehr
   - Assembly: Montage, Baugruppen
   - API: Programmierschnittstellen, externe Zugriffe

2. KUMAVISION Base (KBA):
   - UDI: Unique Device Identification, Medizinprodukte-Kennzeichnung
   - Call: Service-Calls, Außendienst-Einsätze
   - LIF: Lieferantenmanagement, Lieferantenintegration
   - OrderQuote: Angebote, Auftragsangebote
   - InventorySummary: Lagerübersicht, Bestandszusammenfassung
   - Common: Basiskomponenten, allgemeine Funktionen (Root für KBA)

3. Healthcare-/MEDTEC-spezifisch:
   - EDocuments: Elektronisches & digitales Dokumentenhandling (z.B. eRechnung, eArztbrief)
   - ECE: Electron Cost Estimation, elektronische Kostenkalkulation
   - MDR: Medical Device Regulation, regulatorische Anforderungen für Medizinprodukte
   - Common: Basiskomponenten, allgemeine Funktionen (Root für HC/MTC)
"""

def ollama_namespace_prompt(object_type: str, object_name: str, filename: str, context: List[Dict], hc_obj: dict, mtc_obj: dict, extension_context: List[Dict] = None, extension_base: str = None) -> str:
    context_str = ""
    for idx, ctx in enumerate(context, 1):
        ctx_type = ctx.get("object_type", "")
        ctx_name = ctx.get("object_name", "")
        ctx_ns = ctx.get("namespace", "")
        ctx_dir = ctx.get("directory", "")
        ctx_content = ctx.get("content", "")[:300].replace("\n", " ")
        context_str += f"\nKontext {idx}: Typ: {ctx_type}, Name: {ctx_name}, Namespace: {ctx_ns}, Verzeichnis: {ctx_dir}, Auszug: {ctx_content}"

    hc_name = hc_obj.get("object_name", "")
    mtc_name = mtc_obj.get("object_name", "")
    hc_info = f"HC-Objektname: {hc_name}" if hc_name else ""
    mtc_info = f"MTC-Objektname: {mtc_name}" if mtc_name else ""
    both_info = ", ".join(filter(None, [hc_info, mtc_info]))

    namespace_rules = (
        "WICHTIG: "
        "Du darfst ausschließlich einen der folgenden Namespaces verwenden und KEINEN anderen: "
        "Warehouse, Utilities, System, Service, Sales, RoleCenters, Purchases, Projects, Profile, Pricing, OtherCapabilities, Manufacturing, Invoicing, Inventory, Integration, HumanResources, Foundation, FixedAssets, Finance, eServices, CRM, CostAccounting, CashFlow, Bank, Assembly, API, "
        "UDI, Call, LIF, OrderQuote, InventorySummary, Common, "
        "EDocuments, ECE, MDR. "
        "Erfinde KEINE neuen Namespaces, verwende KEINE Präfixe oder Kombinationen wie 'HC.MTC.Common', 'HC.MDR' oder ähnliches. "
        "Wenn ein Objekt mehreren Bereichen zugeordnet werden kann oder übergreifend ist, dann wähle ausschließlich den Namespace 'Common' (Root-Ebene). "
        "Jede andere Antwort als einer der oben genannten Namespaces ist FALSCH."
    )

    related_hint = (
        "Hinweis: Wenn es Objekte mit ähnlichem Namen oder sehr ähnlicher Funktionalität gibt (z.B. 'AdditionalFieldEventLib' und 'AdditionalFieldLib'), "
        "sollten diese nach Möglichkeit im gleichen Namespace landen, außer es gibt einen klaren fachlichen Grund für eine Trennung."
    )

    # Prompt für Extension-Objekte
    if extension_context and extension_base:
        ext_ctx = extension_context[0] if extension_context else {}
        ext_ns = ext_ctx.get("namespace", "")
        ext_type = ext_ctx.get("object_type", "")
        ext_name = ext_ctx.get("object_name", "")
        ext_dir = ext_ctx.get("directory", "")
        ext_info = f"Das Objekt erweitert {ext_type} '{ext_name}' (Namespace: {ext_ns}, Verzeichnis: {ext_dir})."
        return (
            f"Du bist ein Experte für Microsoft Dynamics 365 Business Central AL-Entwicklung. "
            f"Für das folgende Extension-Objekt soll ein passender Namespace vorgeschlagen werden. "
            f"{ext_info}\n"
            f"Der Namespace der Extension MUSS sich am Namespace des erweiterten Objekts orientieren. "
            f"{namespace_rules}\n"
            f"{related_hint}\n"
            f"Hier ist eine Übersicht der bislang verfügbaren Namespaces:\n{NAMESPACE_OVERVIEW}\n"
            f"Objekttyp: {object_type}, Objektname: {object_name}, Dateiname: {filename}. {both_info}\n"
            f"Berücksichtige folgende ähnliche Objekte aus Base Application und KBA:{context_str}\n"
            f"Deine Aufgabe:\n"
            f"- Schlage einen passenden Namespace aus obiger Liste vor (bevorzugt den Namespace des erweiterten Objekts).\n"
            f"- Begründe deine Entscheidung.\n"
            f"- Schlage wenn möglich alternative Namespaces aus der Liste vor und begründe diese Alternativen.\n"
            f"Gib das Ergebnis als JSON im folgenden Format zurück:\n"
            f'{{"namespace": "...", "reason": "...", "alternatives": [{{"namespace": "...", "reason": "..."}}]}}'
        )
    # Prompt für andere Objekte
    else:
        return (
            f"Du bist ein Experte für Microsoft Dynamics 365 Business Central AL-Entwicklung. "
            f"Für das folgende Objekt aus den Lösungen HC (Healthcare) und/oder MTC (Medtec) soll ein passender Namespace vorgeschlagen werden. "
            f"Beachte: Beide Apps stammen aus einem gemeinsamen Ursprung und haben eine Abhängigkeit zu KUMAVISION Base (KBA) und Microsoft Base Application. "
            f"Die KBA hat noch keine Namespaces, aber das Verzeichnis gibt einen Hinweis auf den zukünftigen Namespace. "
            f"Die Base Application verwendet bereits Namespaces, die möglichst eingehalten werden sollten. "
            f"{namespace_rules}\n"
            f"{related_hint}\n"
            f"Hier ist eine Übersicht der bislang verfügbaren Namespaces:\n{NAMESPACE_OVERVIEW}\n"
            f"Objekttyp: {object_type}, Objektname: {object_name}, Dateiname: {filename}. {both_info}\n"
            f"Berücksichtige folgende ähnliche Objekte aus Base Application und KBA:{context_str}\n"
            f"Deine Aufgabe:\n"
            f"- Analysiere den Kernaspekt des Objekts (z.B. Zweck, verwendete Objekte, Schlüsselwörter).\n"
            f"- Prüfe, ob ein existierender Namespace aus der Liste passt oder ob bekannte Namespaces im Objekt verwendet werden.\n"
            f"- Schlage einen passenden Namespace aus obiger Liste vor.\n"
            f"- Begründe deine Entscheidung.\n"
            f"- Schlage wenn möglich alternative Namespaces aus der Liste vor und begründe diese Alternativen.\n"
            f"Gib das Ergebnis als JSON im folgenden Format zurück:\n"
            f'{{"namespace": "...", "reason": "...", "alternatives": [{{"namespace": "...", "reason": "..."}}]}}'
        )

def extract_extended_object(content: str) -> Optional[Tuple[str, str]]:
    """
    Extrahiert das erweiterte Objekt bei Extension-Objekten (z.B. tableextension 50100 "MyTableExt" extends "BaseTable")
    Gibt (Typ, Name) zurück oder None.
    """
    import re
    match = re.search(r'^(tableextension|pageextension|enumextension|reportextension|permissionsetextension)\s+\d+\s+"?([\w\d_]+)"?\s+extends\s+"?([\w\d_]+)"?', content, re.IGNORECASE | re.MULTILINE)
    if match:
        ext_type = match.group(1)
        base_name = match.group(3)
        return ext_type, base_name
    return None

def retrieve_context_for_extension(base_object_name: str, base_object_type: str = None, top_k: int = 1) -> List[Dict]:
    # Suche gezielt nach dem Basisobjekt in LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(LANCEDB_TABLE)
    filter_expr = f"object_name = '{base_object_name}'"
    if base_object_type:
        filter_expr += f" and object_type = '{base_object_type}'"
    try:
        results = table.search(filter=None, where=filter_expr).limit(top_k).to_list()
        return results
    except Exception as e:
        print(f"LanceDB Extension-Retrieval-Fehler: {e}")
        return []

def validate_namespace(ns: str) -> str:
    return ns if ns in ALLOWED_NAMESPACES else "INVALID"

def validate_alternatives(alternatives: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(ns, reason) for ns, reason in alternatives if ns in ALLOWED_NAMESPACES]

def suggest_namespace_ollama(object_type: str, object_name: str, filename: str, hc_obj: dict, mtc_obj: dict) -> Tuple[str, str, List[Tuple[str, str]], str, List[str]]:
    # Hole den Dateipfad für den Content
    filepath = None
    if hc_obj.get("filename"):
        filepath = os.path.join(HC_ROOT, hc_obj["filename"])
    elif mtc_obj.get("filename"):
        filepath = os.path.join(MTC_ROOT, mtc_obj["filename"])
    content = ""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            pass

    # Prüfe auf Extension-Objekt
    ext_info = extract_extended_object(content) if content else None
    extension_context = []
    extension_base = None
    if ext_info:
        ext_type, base_name = ext_info
        extension_base = base_name
        # Suche gezielt nach dem Basisobjekt in LanceDB
        extension_context = retrieve_context_for_extension(base_name)
        # Fallback: Wenn nichts gefunden, Standard-Kontext
        if not extension_context:
            extension_context = retrieve_context(object_type, object_name, top_k=3)
    else:
        extension_context = None

    # Standard-Kontext für alle Objekte
    context = retrieve_context(object_type, object_name, top_k=3)

    prompt = ollama_namespace_prompt(
        object_type,
        object_name,
        filename,
        context,
        hc_obj,
        mtc_obj,
        extension_context=extension_context if ext_info else None,
        extension_base=extension_base
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        import json as pyjson
        text = result.get("response", "")
        print(f"\n--- Ollama Antwort für {object_type} {object_name} ---\n{text}\n")  # Debug-Ausgabe
        match = re.search(r'\{.*\}', text, re.DOTALL)
        considered_namespaces = []
        if match:
            data = pyjson.loads(match.group(0))
            ns = data.get("namespace", "")
            reason = data.get("reason", "")
            alternatives = []
            for alt in data.get("alternatives", []):
                alternatives.append((alt.get("namespace", ""), alt.get("reason", "")))
            # Validierung
            valid_ns = validate_namespace(ns)
            valid_alternatives = validate_alternatives(alternatives)
            # Analyse-Hinweis bei ungültigem Namespace
            if valid_ns == "INVALID":
                reason = f"Ungültiger Namespace '{ns}' vorgeschlagen. Nur folgende sind erlaubt: {', '.join(sorted(ALLOWED_NAMESPACES))}. Grund laut Modell: {reason}"
            considered_namespaces.append(f"Hauptvorschlag: {valid_ns} - {reason}")
            for alt in valid_alternatives:
                considered_namespaces.append(f"Alternative: {alt[0]} - {alt[1]}")
            return valid_ns, reason, valid_alternatives, text, considered_namespaces
        else:
            return "", "Konnte kein JSON aus Ollama-Antwort extrahieren.", [], text, []
    except Exception as e:
        return "", f"Ollama-Fehler: {e}", [], "", []

def read_existing_csv(csv_path: str) -> set:
    existing_keys = set()
    if not os.path.exists(csv_path):
        return existing_keys
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Nutze ObjectType + ObjectName HC + ObjectName MTC als Schlüssel
            key = (
                row.get("ObjectType", "").strip().lower(),
                row.get("ObjectName HC", "").strip().lower(),
                row.get("ObjectName MTC", "").strip().lower()
            )
            existing_keys.add(key)
    return existing_keys

def collect_objects(root: str, prefix: str) -> Dict[str, Dict]:
    result = {}
    for filepath in find_al_files(root):
        info = extract_object_info(filepath)
        if not info:
            continue
        obj_type, obj_name = info
        obj_name_noprefix = remove_prefix(obj_name, prefix)
        # ACHTUNG: Key muss aus obj_type und obj_name_noprefix bestehen, aber obj_name_noprefix muss wirklich NUR das Prefix entfernen!
        # Fehlerquelle: remove_prefix ist case-sensitive und prüft nur auf Großbuchstaben!
        # Besser: Case-insensitive Prefix-Entfernung
        if obj_name.upper().startswith(prefix.upper()):
            obj_name_noprefix = obj_name[len(prefix):]
        else:
            obj_name_noprefix = obj_name
        key = f"{obj_type.lower()}|{obj_name_noprefix.lower()}"
        result[key] = {
            "object_type": obj_type,
            "object_name": obj_name,
            "object_name_noprefix": obj_name_noprefix,
            "filename": os.path.basename(filepath)
        }
    return result

def build_row(key, hc_objs, mtc_objs):
    hc = hc_objs.get(key, {})
    mtc = mtc_objs.get(key, {})

    object_type = hc.get("object_type") or mtc.get("object_type") or ""
    # Hier wird jeweils das Original aus HC und MTC verwendet:
    object_name_hc = hc.get("object_name", "")
    object_name_mtc = mtc.get("object_name", "")
    filename = hc.get("filename") or mtc.get("filename") or ""

    suggest_base = hc or mtc
    ns, ns_reason, alternatives, analyse, considered_namespaces = suggest_namespace_ollama(
        object_type,
        suggest_base.get("object_name_noprefix", ""),
        filename,
        hc,
        mtc
    )
    alt_ns = "; ".join([a[0] for a in alternatives])
    alt_reason = "; ".join([a[1] for a in alternatives])

    considered_str = " | ".join(considered_namespaces)
    analyse_full = f"{analyse.strip().replace(chr(10), ' ')} --- Überlegung der Namespaces: {considered_str}"

    # Schreibe nur, wenn kein JSON-Fehler oder INVALID Namespace
    if ns == "" or ns == "INVALID" or "Konnte kein JSON aus Ollama-Antwort extrahieren." in ns_reason or "Ollama-Fehler" in ns_reason:
        return None

    return {
        "ObjectType": object_type,
        "ObjectName HC": object_name_hc,
        "ObjectName MTC": object_name_mtc,
        "Namespace Vorschlag": ns,
        "Namespace Begründung": ns_reason,
        "Alternative Namespace Vorschlag": alt_ns,
        "Alternative Namespace Begründung": alt_reason,
        "Dateiname": filename,
        "Analyse": analyse_full
    }

def main():
    hc_objs = collect_objects(HC_ROOT, PREFIX_HC)
    mtc_objs = collect_objects(MTC_ROOT, PREFIX_MTC)

    # Matching: gleiche Objekte (gleicher Typ + Name ohne Prefix) sollen im selben Namespace landen
    # Die Schlüssel sind bereits normalisiert (siehe collect_objects)
    all_keys = list(sorted(set(hc_objs.keys()).union(mtc_objs.keys())))

    # Lade bestehende Ergebnisse
    existing_keys = read_existing_csv(CSV_OUTPUT)

    # Keine Begrenzung mehr auf 10 Einträge
    test_keys = all_keys

    # Filtere nur neue Objekte
    filtered_keys = []
    for key in test_keys:
        hc = hc_objs.get(key, {})
        mtc = mtc_objs.get(key, {})
        obj_type = hc.get("object_type") or mtc.get("object_type") or ""
        # Wichtig: Für die CSV-Spalte "ObjectName HC" und "ObjectName MTC" muss jeweils das Original aus hc_objs bzw. mtc_objs verwendet werden!
        obj_name_hc = hc.get("object_name", "")
        obj_name_mtc = mtc.get("object_name", "")
        csv_key = (obj_type.strip().lower(), obj_name_hc.strip().lower(), obj_name_mtc.strip().lower())
        if csv_key not in existing_keys:
            filtered_keys.append(key)

    import time
    start_time = time.time()

    fieldnames = [
        "ObjectType",
        "ObjectName HC",
        "ObjectName MTC",
        "Namespace Vorschlag",
        "Namespace Begründung",
        "Alternative Namespace Vorschlag",
        "Alternative Namespace Begründung",
        "Dateiname",
        "Analyse"
    ]

    write_header = not os.path.exists(CSV_OUTPUT) or os.stat(CSV_OUTPUT).st_size == 0
    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(build_row, key, hc_objs, mtc_objs) for key in filtered_keys]
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Namespace-Vorschläge", unit="Objekt")):
                row = future.result()
                if row is not None:
                    writer.writerow(row)
                    csvfile.flush()
                elapsed = time.time() - start_time
                avg = elapsed / (i + 1) if i + 1 > 0 else 0
                remaining = avg * (len(futures) - (i + 1))
                print(f"Bearbeitet: {i+1}/{len(futures)} | Verstrichen: {elapsed:.1f}s | Ø {avg:.1f}s/Objekt | Rest: {remaining/60:.1f}min")

    # Kein Sammeln und Sortieren im Speicher mehr nötig

if __name__ == "__main__":
    main()
