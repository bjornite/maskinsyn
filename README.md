# Prosjektarbeid i UNIK4690 - Maskinsyn

Vi har laget et system som tracker et objekt.

## Bruk av systemet

Systemet har 3 ulike moduser, man velger modus ved å endre mode-verdien i main.cpp

###Kontrollknapper
q - avslutt programmet <br />
g - reset<br />
x - slett rektangelmodellen til featurematcheren <br />
r - øk terskelen så fargemodellen blir mer generell (default er 0.05)<br />
f - senk terskelen så fargemodellen blir mer spesifikk<br />
e - øker størrelsen på kjernen som brukes til å lukke masken i fargemodellen (default er 1)<br />
d - senker størrelsen på kjernen som brukes til å lukke masken i fargemodellen<br />
w - øker antallet iterasjoner av lukking i fargemodellen(default er 1)<br />
s - senker antallet iterasjoner av lukking i fargemodellen<br />
a - endre mode <br />

###Mode = 1: Featurematching

Systemet genererer en multivariat gaussmodell av fargefordelingen i en firkant midt i nedre halvdel av bildet.

###Mode = 2: Fargematching

Systemet generer en featurebeskrivelse av det første objektet som beveger seg i bildet etter at systemet starter. Ekte objektmatcher blir markert grønne, mens ekstra matcher den finner i rektangelet blir merket gule.

###Mode = 3: En kombinasjon av featurematching og fargematching

Systemet generer både en featurebeskrivelse og en fargemodell av det første objektet som beveger seg i bildet etter at systemet starter.



