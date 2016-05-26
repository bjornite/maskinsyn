# Prosjektarbeid i UNIK4690 - Maskinsyn

Vi har laget et system som tracker et objekt.

## Bruk av systemet

Systemet har 3 ulike moduser, man velger modus ved å endre mode-verdien i main.cpp

###Mode = 1: Featurematching

Systemet generer en featurebeskrivelse av det første objektet som beveger seg i bildet etter at systemet starter.

Kontrollknapper: <br />
g - reset

###Mode = 2: Fargematching

Systemet genererer en multivariat gaussmodell av fargefordelingen i en firkant midt i nedre halvdel av bildet.

Kontrollknapper:
g - reset<br />
r - øk terskelen så modellen blir mer generell (default er 0.05)<br />
f - senk terskelen så modellen blir mer spesifikk<br />
e - øker størrelsen på kjernen som brukes til å lukke masken (default er 1)<br />
d - senker størrelsen på kjernen som brukes til å lukke masken<br />
w - øker antallet iterasjoner av lukking (default er 1)<br />
s - senker antallet iterasjoner av lukking<br />

###Mode = 3: En kombinasjon av featurematching og fargematching

Systemet generer både en featurebeskrivelse og en fargemodell av det første objektet som beveger seg i bildet etter at systemet starter.

Kontrollknapper:<br />
g - reset
