### Projektidee: Automatisierte Namespace-Vergabe für Dynamics Business Central AL-Projekte

#### Ziel
Das Ziel dieses Projekts ist es, für jedes AL-Objekt in den verwandten Lösungen **MEDTEC** und **HC** automatisiert einen passenden Namespace aus einer vorgegebenen Auswahl zu ermitteln. Die Entscheidung soll begründet werden, und es sollen alternative Namespaces mit Erklärungen vorgeschlagen werden. Die Ergebnisse werden in einer CSV-Datei ausgegeben, die in Excel weiterverarbeitet werden kann.

#### Hintergrund
- **MEDTEC** und **HC** sind zwei verwandte Lösungen, die viele gleichnamige Objekte mit unterschiedlichen Präfixen haben (z. B. `KVSMTC` für MEDTEC und `KVSMED` für HC).
- Die Namespaces zwischen beiden Lösungen sollten möglichst synchron sein.
- Es gibt Abhängigkeiten zu anderen mächtigen Apps:
  - **Base Application** (Microsoft Standard)
  - **KUMAVISION Base** (KBA, unterste Schicht)
- Die Namespace-Vergabe orientiert sich an den bestehenden Namespaces der Base Application und KUMAVISION Base.

#### Vorgehen
1. **Datenvorbereitung**:
   - Die Base Application und KUMAVISION Base werden lokal in eine **LanceDB** vektorisiert.
   - Alle `*.al`-Dateien der HC-Lösung werden per Python-Batchjob geladen.
   - Falls ein entsprechendes MEDTEC-Objekt existiert, wird dieses ebenfalls geladen.
   - Alle noch nicht geladenen Objekte von MEDTEC werden ergänzt, sodass alle relevanten Objekte von HC und MEDTEC im Speicher verfügbar sind.

2. **Namespace-Vorschläge durch KI**:
   - Eine KI analysiert jedes Objekt und schlägt einen passenden Namespace vor.
   - Die KI kann dabei auf die vektorisierten Daten der Base Application und KUMAVISION Base zugreifen (mittels Retrieval-Augmented Generation, RAG), um fundierte Entscheidungen zu treffen.
   - Neben dem Hauptvorschlag werden alternative Namespaces mit Begründungen ermittelt.

3. **Ergebnisexport**:
   - Die Ergebnisse werden in einer CSV-Datei gespeichert. Die Datei enthält folgende Spalten:
     - **Objekt-Typ**, **Objekt-ID**, **Objekt-Name**
     - **Namespace-Vorschlag** mit Begründung
     - **Alternative Namespaces** (sortiert) mit Erklärungen

#### Einschränkungen
- Das Projekt ist für eine einmalige Verwendung konzipiert und muss nicht wiederverwendbar sein.
- Die Lösung soll pragmatisch und zweckmäßig umgesetzt werden.

#### Namespace-Übersicht
Die verfügbaren Namespaces stammen aus drei Quellen:
_Anmerkung: Es können auch andere Namesspaces enthalten sein, die auf der List noch nicht enthalten sind.

1. **Microsoft Standard**:
   - `Warehouse`
   - `Utilities`
   - `System`
   - `Service`
   - `Sales`
   - `RoleCenters`
   - `Purchases`
   - `Projects`
   - `Profile`
   - `Pricing`
   - `OtherCapabilities`
   - `Manufacturing`
   - `Invoicing`
   - `Inventory`
   - `Integration`
   - `HumanResources`
   - `Foundation`
   - `FixedAssets`
   - `Finance`
   - `eServices`
   - `CRM`
   - `CostAccounting`
   - `CashFlow`
   - `Bank`
   - `Assembly`
   - `API`

2. **KUMAVISION Base (KBA)**:
   - `UDI`
   - `Call`
   - `LIF`
   - `OrderQuote`
   - `InventorySummary`
   - `Common`  (Der temporäre Namespace dient als Ersatz für den Root Namespace Kumavision.Base. Die entsprechenden Dateien befinden sich im Verzeichnis Common.)

3. **Healthcare-spezifisch (HC) und MEDTEC-spezifisch (MTC)**:
   - `EDocuments`
   - `ECE`
   - `MDR`
   - `Common`  (Der temporäre Namespace dient als Ersatz für den Root Namespace Kumavision.Healthcare bzw. Kumavision.Mtc. Die entsprechenden Dateien befinden sich im Verzeichnis Common.)

#### Besonderheiten
- Einige Objekte bedienen mehrere Namespaces. Hier muss entschieden werden, ob das Objekt aufgeteilt oder einem Root-Element zugeordnet wird.
- Die Root-Namespaces sind:
  - **Kumavision.Healthcare** für HC (Verschiebe-Ziel wird später das Dateisystem-Verzeichnis Common sein)
  - **Kumavision.Mtc** für MEDTEC (Verschiebe-Ziel wird später das Dateisystem-Verzeichnis Common sein)
- Es gibt maximal drei Namespace-Ebenen.

#### Beispielausgabe
| Objekt-Typ    | Objekt-ID | Objekt-Name | Namespace Vorschlag | Begründung                          | Alternative Namespaces (sortiert) mit Erklärung |
|---------------|-----------|-------------|----------------------|--------------------------------------|------------------------------------------------|
| PageExtension | 12345     | KVSMEDFoo   | Invoice             | Weil es etwas mit Rechnungen zu tun hat | - Finance: Verknüpfung mit Finanzdaten         |