# Riduzione della lista cieca a 20 eventi

Regola fissata il 5 novembre 2026, con l'accordo dell'autore, prima di applicarla. Usa solo il tipo, le date e l'id degli eventi di `lista_cieca.csv` (sha256 `ed4db539ff37b2590e5e5947e5f687a46e04f85543d977d51f606c8e9b15ae56`, invariata): non usa le segnalazioni del modello né i giorni più eclatanti.

1. Si tengono gli eventi con date esatte (`affidabilita_date = alta`): 39 su 60.
2. Per ognuno dei cinque tipi se ne prendono quattro, uno per periodo: 2005–2009, 2010–2014, 2015–2019, 2020–2025.
3. In ogni periodo si prende l'evento che inizia più vicino al centro del periodo.
4. Per un periodo senza eventi di quel tipo si prende, fra quelli rimasti dello stesso tipo, quello che inizia più vicino al centro del periodo.
5. A pari distanza vince il primo per `id`.

Applicata con `riduci_lista.py`, che controlla l'hash della lista cieca. Risultato: `lista_ridotta.csv`, 20 eventi (4 per tipo), sha256 `3e0c89c0c5d76382b92d200464dde78ddbe6378d886b9d182f6873098f4ee640`.

Limite: il caldo ha solo quattro eventi con date esatte, tutti fra il 2017 e il 2022, quindi li prende tutti; nel freddo il periodo 2005–2009 non ha eventi con date esatte.

Il file locale ha fine riga Windows (CRLF); la copia nel repository del codice (`docs/era5_casi_noti/lista_ridotta.csv`, fine riga LF, stesso contenuto) ha sha256 `be9f276b65ee391e385e0100dcaedf42270a34971117483408753387e2d77bd4`.

## Lista con le sole date esatte (6 novembre 2026)

Per il documento agli energetici si usa, al posto dei 20, il primo passo della regola da solo: gli eventi di `lista_cieca.csv` (hash invariato) con `affidabilita_date = alta`, 39 su 60 (vento 11, pioggia 11, polvere 8, freddo 5, caldo 4), ordinati per tipo e data di inizio. File `lista_esatta.csv` (fine riga LF, identico alla copia `docs/era5_casi_noti/lista_esatta.csv` del repository del codice), sha256 `1fea23b86f3b31782801fe84fc7c8962cb6ec1d68de17a84e132fa37d48b689e`. La scelta fra 20 e 39 è stata fatta dopo aver letto le quote dei 20, ma i 39 sono definiti dalla sola affidabilità delle date, fissata nella lista cieca.
