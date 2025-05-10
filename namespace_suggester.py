import os
import re
import csv
from typing import Dict, List, Tuple, Optional

HC_ROOT = "./HC"    # Passe ggf. an
MTC_ROOT = "./MTC"  # Passe ggf. an
CSV_OUTPUT = "namespace_suggestions.csv"

OBJECT_PATTERN = re.compile(r'^(table|page|codeunit|report|xmlport|query|enum|interface|controladdin|pageextension|tableextension|enumextension|profile|dotnet|entitlement|permissionset|permissionsetextension|reportextension|enumvalue|entitlementset|entitlementsetextension)\s+(\d+)?\s*"?([\w\d_]+)"?', re.IGNORECASE)

PREFIX_HC = "KVSMED"
PREFIX_MTC = "KVSMTC"

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
    if name.upper().startswith(prefix):
        return name[len(prefix):]
    return name

def suggest_namespace(obj_type: str, obj_name: str, filename: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    # Platzhalter-Logik für Namespace-Vorschlag und Begründung
    # Hier ggf. KI/Heuristik einbinden
    if "invoice" in obj_name.lower():
        ns = "Invoicing"
        reason = "Objektname enthält 'invoice', daher Invoicing."
        alternatives = [("Finance", "Verknüpfung mit Finanzdaten"), ("Sales", "Verbindung zu Verkaufsprozessen")]
    elif "customer" in obj_name.lower():
        ns = "CRM"
        reason = "Objektname enthält 'customer', daher CRM."
        alternatives = [("Sales", "Kundenbezug im Verkauf"), ("Foundation", "Allgemeiner Kundenstamm")]
    else:
        ns = "Common"
        reason = "Kein spezifisches Schlüsselwort gefunden, daher Common."
        alternatives = [("OtherCapabilities", "Allgemeine Funktionalität"), ("Utilities", "Hilfsobjekt")]
    return ns, reason, alternatives

def collect_objects(root: str, prefix: str) -> Dict[str, Dict]:
    result = {}
    for filepath in find_al_files(root):
        info = extract_object_info(filepath)
        if not info:
            continue
        obj_type, obj_name = info
        obj_name_noprefix = remove_prefix(obj_name, prefix)
        key = f"{obj_type.lower()}|{obj_name_noprefix.lower()}"
        result[key] = {
            "object_type": obj_type,
            "object_name": obj_name,
            "object_name_noprefix": obj_name_noprefix,
            "filename": os.path.basename(filepath)
        }
    return result

def main():
    hc_objs = collect_objects(HC_ROOT, PREFIX_HC)
    mtc_objs = collect_objects(MTC_ROOT, PREFIX_MTC)

    all_keys = set(hc_objs.keys()).union(mtc_objs.keys())
    rows = []

    for key in sorted(all_keys):
        hc = hc_objs.get(key, {})
        mtc = mtc_objs.get(key, {})

        object_type = hc.get("object_type") or mtc.get("object_type") or ""
        object_name_hc = hc.get("object_name", "")
        object_name_mtc = mtc.get("object_name", "")
        filename = hc.get("filename") or mtc.get("filename") or ""

        # Für Namespace-Vorschlag: Nutze bevorzugt HC, sonst MTC
        suggest_base = hc or mtc
        ns, ns_reason, alternatives = suggest_namespace(
            object_type,
            suggest_base.get("object_name_noprefix", ""),
            filename
        )
        alt_ns = "; ".join([a[0] for a in alternatives])
        alt_reason = "; ".join([a[1] for a in alternatives])

        rows.append({
            "ObjectType": object_type,
            "ObjectName HC": object_name_hc,
            "ObjectName MTC": object_name_mtc,
            "Namespace Vorschlag": ns,
            "Namespace Begründung": ns_reason,
            "Alternative Namespace Vorschlag": alt_ns,
            "Alternative Namespace Begründung": alt_reason,
            "Dateiname": filename
        })

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "ObjectType",
            "ObjectName HC",
            "ObjectName MTC",
            "Namespace Vorschlag",
            "Namespace Begründung",
            "Alternative Namespace Vorschlag",
            "Alternative Namespace Begründung",
            "Dateiname"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()
