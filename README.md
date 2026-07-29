### Projekt beschreibung

Dieses Projekt beinhaltet die implementierungen des [Doubling-Algorithmus](https://doi.org/10.1145/258533.258657) sowie die [verbesserung durch Parallelisierung](https://doi.org/10.1007/978-3-540-85363-3_14) für das streaming k-Center Problem. Diese Algorithmen werden verwendet um die Ergebnisse mit denen des [Gonzalez-Algorithmus](<https://doi.org/10.1016/0304-3975(85)90224-5>) für das k-Center Problem zu vergleichen.

Es sind mehrere vergleichende Experimente vorhanden, welche in der main.py ausgeführt werden können. Ebenso können die Ergebnisse Visualisiert werden indem die jeweilige plot Funktion in der main.py ausgeführt wird.

### Voraussetzungen

- Python 3.10 oder neuer
- `pip`

### 1. Repository klonen

```bash
git clone <repository-url>
cd <projektname>
```

### 2. Virtuelle Umgebung erstellen

##### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

##### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

##### Windows (Eingabeaufforderung)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Abhängigkeiten installieren

Nach dem Aktivieren der virtuellen Umgebung alle benötigten Pakete installieren:

```bash
pip install -r requirements.txt
```

### 4. Anwendung starten

```bash
python main.py
```
