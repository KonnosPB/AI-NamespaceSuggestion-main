import lancedb
from collections import defaultdict
import os
import requests
import random

LANCEDB_PATH = "./lancedb"
LANCEDB_TABLE = "namespace_vectors"
OUTPUT_FILE = "namespace_definitions.md"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi4:latest"
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest")

NAMESPACE_HINTS = """
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

def ollama_namespace_definition_prompt(namespace, examples):
    example_str = "\n".join(
        f"- {obj.get('object_type','')} {obj.get('object_name','')} ({obj.get('filename','')})"
        for obj in examples
    )
    prompt = (
        f"Du bist ein Experte für Microsoft Dynamics 365 Business Central AL-Entwicklung. "
        f"Deine Aufgabe ist es, für den Namespace '{namespace}' eine prägnante, KI-geeignete Definition zu erstellen, "
        f"die einer anderen KI hilft, nach einer Analyse von AL-Quellcode zu entscheiden, ob ein Objekt diesem Namespace zugeordnet werden sollte.\n"
        f"Nutze folgende Hinweise zu den verfügbaren Namespaces:\n{NAMESPACE_HINTS}\n"
        f"Berücksichtige typische Zwecke, Domänen, Objektarten, Schlüsselbegriffe und die Abgrenzung zu anderen Namespaces. "
        f"Formuliere die Definition so, dass sie als Entscheidungsregel für eine KI dienen kann.\n"
        f"Beispielobjekte für diesen Namespace:\n{example_str}\n"
        f"Gib die Definition als Fließtext auf Deutsch zurück."
    )
    return prompt

def get_namespace_definition(namespace, examples):
    prompt = ollama_namespace_definition_prompt(namespace, examples)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"Fehler bei OLLAMA: {e}"

def main():
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(LANCEDB_TABLE)
    df = table.to_pandas()
    ns_groups = defaultdict(list)
    for _, row in df.iterrows():
        ns = row.get("namespace", "")
        if ns:
            ns_groups[ns].append(row)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ns, objs in ns_groups.items():
            # Ziehe eine kleine, zufällige Auswahl von Beispielobjekten (max 5)
            examples = random.sample(objs, min(5, len(objs)))
            definition = get_namespace_definition(ns, examples)
            f.write(f"## Namespace: {ns}\n")
            f.write(f"**Definition:**\n{definition}\n")
            f.write(f"**Beispielobjekte:**\n")
            for obj in examples:
                f.write(f"- {obj.get('object_type','')} {obj.get('object_name','')} ({obj.get('filename','')})\n")
            f.write("\n")
    print(f"Namespace-Definitionen nach {OUTPUT_FILE} geschrieben.")

if __name__ == "__main__":
    main()
