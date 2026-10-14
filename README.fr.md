# MetroPT Anomaly Benchmark — Détecter les fuites d'air sur le compresseur d'un métro

[![tests](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Français**

Comparaison de méthodes de détection d'anomalies sur des **données de capteurs industriels réelles** : 7 mois de mesures de l'unité de production d'air d'un train du métro de Porto, avec quatre pannes documentées (fuites d'air). Les méthodes vont des règles de contrôle statistique au machine learning classique et au deep learning. Toutes tournent sur les mêmes données, avec le même protocole d'évaluation.

**Question** : quelle méthode détecte les fuites le plus tôt, avec le moins de fausses alarmes, et la complexité supplémentaire en vaut-elle la peine ?

> **État** : en cours. Notebooks 01–05 terminés (données, protocole, références statistiques, ML classique, méthodes temporelles) ; deep learning et comparaison finale à venir.

## Premiers résultats

| | |
|---|---|
| **Signature d'une panne** | Pendant une fuite, le compresseur ne cycle plus et reste **en charge 99,5 % du temps** (contre 15 % normalement), avec une huile **~13 °C plus chaude**. C'est une anomalie *collective* : chaque mesure reste dans la plage normale, la détection doit donc s'appuyer sur des fenêtres de temps et non sur des mesures isolées |
| **Alarme existante** | Le signal de basse pression embarqué ne se déclenche que sur 2,4 % des mesures en panne |
| **Qualité des données** | Une mesure toutes les ~10 s (et non 1 Hz comme documenté) ; 18 % de la période manquante ; 7 jours de valeurs de capteurs figées à exclure |
| **Vérité terrain** | 8 épisodes de charge continue ≥ 1 h ne figurent **pas** dans les rapports de panne : les étiquettes sont incomplètes. L'évaluation se fait donc par événement, et les épisodes non déclarés sont présentés à part |
| **Alerte précoce** | Avant la panne 4, le compresseur était déjà surchargé depuis 14 heures : détecter avant le début déclaré paraît possible |
| **Des fenêtres, pas des mesures** | Avec le protocole d'évaluation, le courant moteur brut détecte 0 panne sur 4 ; sa moyenne sur 2 h en détecte 4/4 avec 0,9 fausse alarme par semaine, dont la panne 4 avec 13,7 h d'avance |
| **Dérive** | Un seuil de température d'huile calibré en hiver donne 1,4 fausse alarme par semaine en été, là où un seuil bien placé n'en donnerait aucune |
| **Meilleures références statistiques** | z-score de la température d'huile par rapport aux 7 jours précédents : 4/4 pannes, 0,28 fausse alarme par semaine. CUSUM sur le courant moteur : 4/4, 0,64 par semaine, panne 4 signalée 12,9 h en avance. Les distances multivariées (max \|z\|, T² de Hotelling) échouent, noyées par une variable à queue lourde |
| **ML classique** | LOF sur les variables de fenêtre égale les meilleures cartes de contrôle (4/4 avec toutes les graines, 0,55 fausse alarme par semaine, panne 4 signalée 12,6 h en avance) sans les battre ; le One-Class SVM détecte 2/4. Les résultats varient jusqu'au triple selon la graine aléatoire : chaque résultat aléatoire est une médiane sur cinq. L'ACP apporte ce que les scores n'ont pas : les signaux qui expliquent chaque alarme |
| **Référence contaminée** | Le Matrix Profile trouve le plus proche voisin de chaque panne dans un épisode de fuite non déclaré du 29 mars, et détecte 0/4. Retirer les épisodes que la règle d'expert repère dans l'apprentissage le fait passer à 4/4 avec 0,71 fausse alarme par semaine. Le même nettoyage fait chuter LOF de 4/4 à 2/4 : son succès reposait en partie sur cette fuite non déclarée. Les résidus de prévision détectent le début des fuites, pas les fuites, avec 2 à 6 fausses alarmes par semaine |

![Signature des pannes](docs/figures/01_failures.png)

## Démarche

| Notebook | Contenu | État |
|---|---|---|
| [01 — Données](notebooks/01_data_exploration.ipynb) | Capteurs, échantillonnage, trous, valeurs figées, cycle de fonctionnement, signature des pannes, complétude des étiquettes | ✅ |
| [02 — Protocole d'évaluation](notebooks/02_evaluation_protocol.ipynb) | Grille à la minute, 18 variables sur fenêtres glissantes, découpage chronologique, évaluation par événement, calibration du seuil, niveau de chance | ✅ |
| [03 — Références statistiques](notebooks/03_statistical_baselines.ipynb) | Règle d'expert, z-score robuste, T² de Hotelling, cartes EWMA et CUSUM, z-score à référence glissante | ✅ |
| [04 — ML classique](notebooks/04_classical_ml.ipynb) | Isolation Forest, One-Class SVM, Local Outlier Factor, reconstruction par ACP ; variables brutes ou corrigées de la dérive ; variabilité selon la graine ; explication des alarmes | ✅ |
| [05 — Méthodes temporelles](notebooks/05_time_series_methods.ipynb) | Résidus de prévision (gradient boosting), Matrix Profile (brut et z-normalisé, référence complète ou nettoyée), contrôle croisé du nettoyage sur LOF | ✅ |
| 06 — Deep learning | Autoencodeurs denses et LSTM | prévu |
| 07 — Résultats | Tableau comparatif, coût de la complexité, recommandation | prévu |

Les notebooks et figures sont rédigés en anglais.

## Données

**MetroPT-3**, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) (CC BY 4.0). Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. *The MetroPT dataset for predictive maintenance.* Scientific Data 9, 764 (2022).

1,5 million de mesures de février à août 2020, sur 15 signaux :

| Type | Signaux |
|---|---|
| Analogiques (7) | `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs` (pressions, bar), `Oil_temperature` (°C), `Motor_current` (A) |
| Numériques (8) | `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` |

Étiquettes : quatre fenêtres de fuite d'air issues des rapports de panne de l'exploitant.

## Installation et reproduction

```bash
git clone https://github.com/mimouniahmed/metro-anomaly-benchmark.git
cd metro-anomaly-benchmark
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest
```

Le jeu de données (218 Mo) est téléchargé automatiquement à la première utilisation par `metrobench.data.load()`, puis mis en cache au format Parquet (22 Mo) dans `data/`.

## Structure

```
metro-anomaly-benchmark/
├── src/metrobench/
│   ├── data.py         # téléchargement, cache Parquet, trous, valeurs figées, séquences, étiquettes
│   ├── features.py     # grille à la minute et variables sur fenêtres glissantes
│   ├── evaluation.py   # découpage temporel, événements d'alarme, métriques, calibration du seuil
│   ├── baselines.py    # z-score robuste, T² de Hotelling, EWMA, CUSUM, z-score à référence glissante
│   ├── ml.py           # Isolation Forest, One-Class SVM, LOF, ACP, variables corrigées de la dérive
│   ├── timeseries.py   # résidus de prévision, Matrix Profile avec nettoyage optionnel de la référence
│   ├── deep.py         # autoencodeurs dense et LSTM (PyTorch)
│   └── plotstyle.py    # style commun des figures
├── notebooks/          # analyse, 01 → 07
├── results/            # tableaux de résultats par notebook (CSV)
├── tests/              # tests unitaires
├── docs/figures/
└── .github/workflows/  # intégration continue
```

## Outils

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · stumpy · PyTorch · Matplotlib · pytest · GitHub Actions

## Auteur

**Ahmed Mimouni** — [github.com/mimouniahmed](https://github.com/mimouniahmed)

## Licence

Publié sous [licence MIT](LICENSE). Le jeu de données MetroPT-3 est distribué par l'UCI sous licence CC BY 4.0 et n'est pas redistribué dans ce dépôt.
