# Text Classification Problem

## Context
In the scope of *DARGMINTS* project, an annotation project was carried out which consisted of annotating argumentation structures in opinion articles published in *Publico* newspaper.

Annotations include several layers:
- Selects text spans that have an argumentative role (e.g. premises or conclusions of arguments) - **Argumentative Discourse Units (ADU)**
- Connect the *ADU*s through ***support*** or ***attack*** relations
- Classify the propositional content of ADUs as:
  - **Proposition of fact**
  - **Proposition of policy**
  - **Proposition of value** - further distinguished with *positive* (+) or *negative* (-) connotation

### Proposition of Fact
Corresponds to pieces of information that can be checked for **truthness**

### Proposition of Policy
Prescribes or suggests a **line of action**, often mentioning **agents** or **entities** capable of carrying such policies

### Proposition of Value
Denote value judgements with strong **subjective nature**, often also have a **polarity** attached, either *positive* or *negative*

## Objective
Build a classifier for these types of ADUs

## Resources
- File containing content of each annotated ADU spain and its 5-class classification
  - **Value**
  - **Value (+)**
  - **Value (-)**
  - **Fact**
  - **Policy**
- Each ADU also has its annotator and document from which it has been taken
- File containing details for each opinion article that has been annotated, including full article content
- Besides ADU content, can make use of any contextual information provided in the corresponding opinion article
- Each opinion article is annotated by 3 different annotators
- One ADU can be annotated by more than one annotator (that may not agree on type of proposition)
