Melting-Point-Project: Amadeus Wasner, Arina Klein, Enrico Fritz

Installation Guide:
Es müssen folgende Pakete in der virtuellen py Umgebung installiert werden:
-> pip install tensorflow xgboost matplotlib pandas numpy scikit-learn rdkit

Relevante Skripts:
->residual.ipynb (Modell) ist am aktuellsten und am meisten performant
->pipeline.py Feature Engineering Funktionen

Der Vollständigkeit halber:
->Neural_Network_clean.ipynb erster Versuch mit NN
->xgboostTest.ipynb erster Versuch mit decision trees

CSV Dateien:
->melting-point-data/train.csv Rohdatei mit vorgefertigten Features von Kaggle
->csv/trainExtended_morgan.csv Trainingsdaten mit unseren Features
->melting-point-data/test.csv Testdatensatz von Kaggle (ohne Tm)