# State of the Art
## Technologien und Ansätze in der KI-gestützten Dokumentationsverarbeitung

### Large Language Models

#### Fortschrittliche Modelle
**GPT-4** von OpenAI repräsentiert den derzeitigen Spitzenstand in der Verarbeitung natürlicher Sprache (NLP). Mit Milliarden von Parametern zählt es zu den leistungsfähigsten Sprachmodellen. Es unterstützt zudem multimodale Eingaben, was bedeutet, dass es nicht nur Text, sondern auch Bildinformationen verarbeiten kann. Diese Vielseitigkeit macht GPT-4 besonders geeignet für High-End-Anwendungen wie persönliche Assistenzsysteme, Übersetzungsdienste oder kreative Content-Erstellung. Allerdings hat diese Leistung ihren Preis: GPT-4 ist extrem ressourcenintensiv und benötigt spezialisierte Hardware wie GPUs oder TPUs. Dies macht den Betrieb teuer und für kleinere Projekte oft unzugänglich.

Ein effizienteres und flexibleres Modell stellt **LLaMA** von Meta dar. LLaMA wurde speziell entwickelt, um leistungsstarke Sprachverarbeitung bei geringeren Ressourcenanforderungen bereitzustellen. Obwohl es weniger Parameter besitzt als GPT-4, zeigt es in vielen NLP-Aufgaben eine vergleichbare Leistung. Besonders in der Forschung und in ressourcenarmen Umgebungen wird LLaMA geschätzt, da es sich einfach anpassen lässt und weniger Rechenleistung benötigt. Diese Eigenschaften machen es ideal für Projekte, bei denen Effizienz und Zugänglichkeit im Vordergrund stehen.

Einen anderen Weg schlagen **GPT-Neo** und **GPT-J** ein, Open-Source-Modelle, die von der EleutherAI-Gemeinschaft entwickelt wurden. Sie bieten Entwicklern die Möglichkeit, leistungsfähige Sprachmodelle zu nutzen, ohne die Beschränkungen proprietärer Systeme wie GPT-4 in Kauf nehmen zu müssen. GPT-Neo ist ein direkter Konkurrent zu GPT-3 und kann viele ähnliche Aufgaben bewältigen, ist jedoch weniger leistungsstark in der Verarbeitung hochkomplexer Anfragen. GPT-J, mit 6 Milliarden Parametern, bietet eine solide Balance zwischen Leistung und Effizienz. Beide Modelle sind frei zugänglich und anpassbar, was sie besonders attraktiv für Forschung, Bildung und kleine Unternehmen macht.

#### Herausforderungen
- **Rechenintensität:** Viele dieser Modelle sind extrem leistungsstark, benötigen jedoch erhebliche Hardware-Ressourcen.
- **Anwendungsabhängigkeit:** LLMs neigen zu sogenannten Halluzinationen, bei denen sie Informationen generieren, die nicht auf den bereitgestellten Daten basieren.

---

## Dokumentationsassistenzsysteme

### Aktuelle Systeme
Aktuelle Systeme im Bereich der KI-gestützten Dokumentationsverarbeitung setzen Maßstäbe für den Zugang und die Interaktion mit technischen Dokumentationen. 

- **NotebookLM:** Von Google AI entwickelt, erleichtert dieses Tool Nutzern den Zugang zu umfangreichen Dokumentationen. Es beantwortet gezielt Fragen, bietet prägnante Zusammenfassungen und hilft dabei, durch komplexe Inhalte zu navigieren.
- **ReadTheDocs AI Assistant:** Entwickelt, um die Nutzung von ReadTheDocs-Dokumentationen zu erleichtern, indem es spezifische Fragen zu Funktionen oder Modulen beantwortet und gezielte Unterstützung bei der Arbeit mit technischen Inhalten bietet.
- **GitHub Copilot:** Obwohl primär als Programmierhilfe konzipiert, generiert es Dokumentationsbeispiele, erklärt relevante Kontexte und erleichtert damit das Verständnis von Programmcode und Dokumentation.

#### Herausforderungen
- Diese Systeme setzen häufig auf leistungsstarke Modelle (z. B. GPT-4) und benötigen teure Cloud-Infrastrukturen.
- Die Abhängigkeit von generativen Modellen birgt das Risiko ungenauer oder erfundener Antworten.

---

## Methods
### AI Algorithms and Architectures (Hyperparameter)

#### Architektur - ChatGPT2
Die Architektur von GPT-2 bildet den Kern des Projekts und wurde gezielt angepasst, um die Verarbeitung technischer Dokumentationen zu gestalten.

1. **Transformer-Decoder-Architektur:** GPT-2 basiert vollständig auf einer autoregressiven Transformer-Decoder-Architektur, die Tokens sequentiell verarbeitet und den nächsten Token basierend auf den vorherigen vorhersagt. Dies macht das Modell ideal für Textgenerierung und Kontextverständnis.

   **Projektspezifische Anpassung:**
   - Eingabedaten werden durch Vorverarbeitung und Chunking in kleinere, kohärente Abschnitte unterteilt, die das maximale Token-Limit nicht überschreiten. Dadurch kann das Modell relevante Kontexte optimal erfassen.

2. **Maximale Eingabelänge:** GPT-2 kann pro Eingabe maximal 1024 Tokens verarbeiten, was für längere technische Dokumentationen eine Herausforderung darstellt.

   **Lösung im Projekt:**
   - **Relevanzbewertung:** Mit einer Log-Likelihood-Analyse (likelihood.py) werden nur die relevantesten Textteile basierend auf der Eingabepriorisierung (z. B. Titel) ausgewählt.
   - **Chunking:** Lange Dokumentationen werden in kleinere Abschnitte zerlegt, die separat verarbeitet werden können.

3. **Token- und Positionsembeddings:** Die Transformer-Architektur nutzt Token- und Positionsembeddings, um die Bedeutung und Reihenfolge von Wörtern zu kodieren. Dadurch kann GPT-2 semantische und strukturelle Zusammenhänge in Texten verstehen.

   **Vorteil für das Projekt:** 
   - Logische Gliederung der Eingabedaten (z. B. Titel und Text) verbessert das Modellverständnis und unterstützt die Kontextgenerierung.

4. **Halluzinationsvermeidung** 
GPT-2 kann, wie viele Sprachmodelle, Informationen halluzinieren.

    **Ansatz im Projekt:**
    - **Prompt Engineering:** Vorgefertigte Prompts strukturieren die Eingaben so, dass Antworten explizit auf die Dokumentation verweisen.
    - **Relevanzfilter:** Vor der Modellnutzung wird sichergestellt, dass nur relevante Inhalte als Input dienen.

---

5. **Workflow-Integration**
Die Architektur von GPT-2 wurde so in das Projekt integriert, dass sie nahtlos mit Vorverarbeitung, Relevanzbewertung und Texterzeugung arbeitet. 

    **Ein Workflow umfasst:**
    1. **Datenbeschaffung:** Thematische Artikel aus Wikipedia.
    2. **Vorverarbeitung:** Bereinigung und Reduzierung auf wesentliche Inhalte.
    3. **Relevanzbewertung:** Priorisierung relevanter Abschnitte.
    4. **Texterzeugung:** Kontextuelle Antworten auf Benutzerprompts.

---

## Data Preprocessing

### 1. Datenbeschaffung
Die Rohdaten stammen aus Wikipedia:
```python
articles = fetch_wikipedia_articles("History", max_articles=20, language="en")
```

### 2. Bereinigung der Rohdaten
Die Rohdaten von Wikipedia kommen mit diversen Sonderzeichen und im Text integrierten Befehlen. 
Die Bereinigung wird durch das Skript `clean.py` durchgeführt, das irrelevante Zeichen entfernt und den Text in ein einheitliches Format bringt. Ziel ist es, die Lesbarkeit und Verarbeitbarkeit der Texte zu verbessern.

```python
# Entferne überflüssige Whitespaces
text = re.sub(r"\s+", " ", text)
# Entferne Sonderzeichen (außer Satzzeichen)
text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)
# Trimme Leerzeichen am Anfang/Ende
return text.strip()
```

### 3. Relevanzbewertung
Die Funktionalitäten aus `likelihood.py` werden genutzt, um die Relevanz von Artikeln anhand der Titel im Kontext eines spezifischen Prompts zu bewerten. Das geschieht durch die Berechnung des Log-Likelihood-Scores mit GPT-2.

```python
# Prompt definieren
prompt = "Describe historical events:"
# Titel-basiertes Filtern durchführen
top_n = 5
filtered_articles = filter_titles_by_prompt(articles, prompt, top_n=top_n)
```

## 4. Speicherung der bereinigten Daten
Bereinigte Artikel werden in einer standardisierten JSON-Struktur gespeichert, die leicht für die weitere Verarbeitung und das Training verwendet werden kann.

```python
print(f"[INFO] Speichere bereinigte Daten in {output_file} ...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)
```

## 5. Statistische Auswertung
Während der Vorverarbeitung werden Statistiken über die Zeichenanzahl berechnet, um die Effizienz der Bereinigung zu überwachen.

```python
total_original_length += original_length
total_cleaned_length += cleaned_length

total_removed_characters = total_original_length - total_cleaned_length
print(f"[INFO] Alle Artikel wurden erfolgreich bereinigt: {len(articles)} Artikel.")
print(f"    [INFO] Ursprüngliche Gesamtzeichenzahl: {total_original_length}")
print(f"    [INFO] Bereinigte Gesamtzeichenzahl: {total_cleaned_length}")
print(f"    [INFO] Insgesamt entfernte Zeichen: {total_removed_characters}")
```

# Hyperparameter

- **temperature: 1.0**  
  Kontrolliert die Kreativität. Niedrige Werte (<1.0) erzeugen präzisere, aber monotone Antworten. Höhere Werte (>1.0) fördern kreative Antworten, können aber unsinnige Inhalte produzieren. Der Wert 1.0 bietet eine ausgewogene Mischung aus Präzision und Kreativität.

- **maxLength: 100**  
  Legt die maximale Länge der generierten Antwort in Tokens fest. Dies begrenzt die Ausgabe auf etwa 20–30 Wörter, ideal für prägnante Antworten, ohne unnötige Rechenressourcen zu verbrauchen.

- **top_k: 50**  
  Begrenzt die Auswahl der nächsten Token auf die 50 wahrscheinlichsten, um kohärente, aber nicht zu monotone Texte zu erzeugen.

- **top_p: 0.9**  
  Nutzt Nucleus Sampling, indem nur Token berücksichtigt werden, deren kumulative Wahrscheinlichkeit 0.9 erreicht. Dadurch wird eine Balance zwischen Vielfalt und Präzision erzielt.

- **repetitionPenalty: 1.2**  
  Bestrafung für wiederholte Wörter oder Phrasen, um redundante Texte zu vermeiden und die Lesbarkeit zu verbessern.

- **doSample: True**  
  Aktiviert das Sampling, sodass die nächsten Token probabilistisch und nicht deterministisch ausgewählt werden, was zu natürlicheren und dynamischen Texten führt.

### Beispielhafte Einstellung der Hyperparameter im Code:

```python
self.temperature = 1.0
self.maxLength = 100
self.top_k = 50
self.top_p = 0.9
self.repetitionPenalty = 1.2
self.doSample = True
```


## Training Process
Details wie das Model trainiert wird.

---

## Model Evaluation
Methoden, um die Performance des Modells zu bewerten: **LIME**?

---

# Performance Comparison and Discussion

- **Results:**  
  Vergleich von 2 verschiedenen Hyperparametereinstellungen.  
  Oder generell: Vergleich von **ChatGPT2** und **ChatGPT4**.

---

# Discussion

## Nutzung leistungsstarker Modelle
Zu Beginn des Projekts wurde untersucht, ob leistungsstärkere Modelle wie **GPT-3** oder **LLaMA** genutzt werden können, um die Zielgenauigkeit und Antwortqualität zu verbessern. Diese Modelle bieten eine größere Kapazität, unterstützen längere Kontexte und könnten theoretisch relevantere Antworten generieren. Allerdings zeigte sich schnell, dass diese Modelle für unsere Hardware nicht praktikabel sind:

- **Lange Ladezeiten:**  
  Die Initialisierung und Verarbeitung selbst kleiner Inputs nahm unverhältnismäßig viel Zeit in Anspruch.
  
- **Hohe Hardwareanforderungen:**  
  Der Betrieb dieser Modelle erfordert spezialisierte GPUs oder Cloud-Infrastrukturen, die in unserem Projekt nicht verfügbar sind.

## Einsatz von GPT-2
Angesichts der Einschränkungen entschieden wir uns für **GPT-2**, ein Modell mit geringeren Ressourcenanforderungen. Diese Entscheidung brachte sowohl Vorteile als auch Herausforderungen mit sich:

### Vorteile:
- **Leichtgewichtig:**  
  GPT-2 ist leichtgewichtig und kann auf Standard-Hardware ausgeführt werden.
  
- **Integration:**  
  Es ermöglicht eine schnelle Integration in Python-Skripte mit der Hugging Face-Bibliothek.

### Herausforderungen:
- **Begrenzte Eingabelänge:**  
  GPT-2 verarbeitet maximal 1024 Tokens, was besonders bei langen technischen Dokumentationen eine Einschränkung darstellt.
  
- **Geringere Modellkapazität:**  
  Im Vergleich zu modernen LLMs wie GPT-4 ist GPT-2 weniger leistungsfähig und kontextsensitiv.

Da die Anzahl der Inputs entscheidend ist, mussten wir durch **Preprocessing** und **Relevanzbewertung** sicherstellen, dass die wichtigsten Inhalte priorisiert werden. Ein Wort entspricht dabei im Durchschnitt **2–3 Tokens**, wodurch komplexere Kontexte oft stark gekürzt werden mussten.

---

# Zielsetzung und Fokus
Unser Projekt war von Anfang an nicht darauf ausgelegt, die besten Ergebnisse zu erzielen oder modernste Modelle zu verwenden. Vielmehr lag der Fokus darauf, den Ablauf und die Methodik der KI-basierten Dokumentationsverarbeitung aufzuzeigen:

- **Demonstration des Prozesses:**  
  Der Workflow von der Datenbeschaffung über das Preprocessing bis zur Modellausgabe sollte nachvollziehbar und funktional sein.

- **Einsatz vorhandener Ressourcen:**  
  Mit unserem Equipment und GPT-2 haben wir ein funktionsfähiges System erstellt, das in der Lage ist, grundlegende Anforderungen zu erfüllen.



