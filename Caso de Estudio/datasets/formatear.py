PrimeraFila = "id ccf age sex painloc painexer relrest pncaden cp trestbps htn chol smoke cigs years fbs dm famhist restecg ekgmo ekgday ekgyr dig prop nitr pro diuretic proto thaldur thaltime met thalach thalrest tpeakbps tpeakbpd dummy trestbpd exang xhypo oldpeak slope rldv5 rldv5e ca restckm exerckm restef restwm exeref exerwm thal thalsev thalpul earlobe cmo cday cyr num lmt ladprox laddist diag cxmain ramus om1 om2 rcaprox rcadist lvx1 lvx2 lvx3 lvx4 lvf cathef junk name\n"
archivo = open("cleveland.data", 'r', errors='ignore')
escritor = open("cleveland2.data", 'w')
escritor.write(PrimeraFila)
aEscribir = ""
for fila in archivo:
	aEscribir += fila 
	if ("name" in fila):
		escritor.write(aEscribir)
		aEscribir = ""
	else:
		aEscribir = aEscribir.rstrip()
		aEscribir += " "
archivo.close()
escritor.close()