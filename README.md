# 18-ha-2010-pj
Demo-Code zu "Wettbewerb künstliche Intelligenz in der Medizin" SoSe 2021

## Erste Schritte

1. cd /Verzeichnis 
2. conda create --name test python=3.8 
3. conda activate test
(* zusätzlich conda install -c anaconda urllib3 , falls Probleme auftreten. Anaconda sollte auf neustem Stand sein.)
4. pip install -r requirements.txt
5. python train.py ausführen, "model_trained" zu erzeugen.
## Wichtig!

Die Dateien 
- predict_pretrained.py
- predict_trained.py
- wettbewerb.py
- score.py



werden von uns beim testen auf den ursprünglichen Stand zurückgesetzt. Es ist deshalb nicht empfehlenswert diese zu verändern.

Bitte alle verwendeten packages in "requirements.txt" bei der Abgabe zur Evaluation angeben. Wir selbst verwenden Python 3.8. Wenn es ein Paket gibt, welches nur unter einer anderen Version funktioniert ist das auch in Ordung. In dem Fall bitte Python-Version mit angeben.
