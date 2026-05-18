import spacy

TEXTO = (
    "Netflix ha encontrado en el juego del calamar su nuevo fenómeno mundial "
    "ni siquiera en la propia plataforma contaban con ello como seguro que "
    "tampoco esperaban recibir multitud de quejas por una escena del cuarto "
    "episodio sin embargo han estado rápidos para responder a la indignación "
    "del público y ha introducido un cambio en el equipo"
)

nlp = spacy.load("es_core_news_sm")
doc = nlp(TEXTO)

SEPARADOR = "=" * 70

# ──────────────────────────────────────────────
# Etapa 1: Segmentación en Frases
# ──────────────────────────────────────────────
print(SEPARADOR)
print("ETAPA 1: SEGMENTACIÓN EN FRASES (Sentence Segmentation)")
print(SEPARADOR)
print()
print("Comentario: El texto original no contiene signos de puntuación.")
print("spaCy utiliza su modelo neuronal basado en el dependency parser para")
print("predecir los límites oracionales a partir de patrones sintácticos y")
print("relaciones de dependencia entre palabras, incluso en ausencia de")
print("puntos, comas u otros delimitadores explícitos.")
print()
for i, sent in enumerate(doc.sents, 1):
    print(f"  Frase {i}: {sent.text}")
print()

# ──────────────────────────────────────────────
# Etapa 2: Tokenización
# ──────────────────────────────────────────────
print(SEPARADOR)
print("ETAPA 2: TOKENIZACIÓN (Tokens)")
print(SEPARADOR)
print()
tokens = [token.text for token in doc]
print(f"  Total de tokens: {len(tokens)}")
print()
print("  Tokens:", tokens)
print()

# ──────────────────────────────────────────────
# Etapa 3: Lematización
# ──────────────────────────────────────────────
print(SEPARADOR)
print("ETAPA 3: LEMATIZACIÓN (Lemmas)")
print(SEPARADOR)
print()
print(f"  {'TOKEN':<20} {'LEMA':<20}")
print(f"  {'-----':<20} {'-----':<20}")
for token in doc:
    print(f"  {token.text:<20} {token.lemma_:<20}")
print()

# ──────────────────────────────────────────────
# Etapa 4: POS Tagging
# ──────────────────────────────────────────────
print(SEPARADOR)
print("ETAPA 4: POS TAGGING (Etiquetado Gramatical)")
print(SEPARADOR)
print()
print(f"  {'TOKEN':<20} {'POS':<12} {'ETIQUETA':<12} {'EXPLICACIÓN'}")
print(f"  {'-----':<20} {'---':<12} {'--------':<12} {'-----------'}")
for token in doc:
    print(f"  {token.text:<20} {token.pos_:<12} {token.tag_:<12} {spacy.explain(token.tag_)}")
print()

# ──────────────────────────────────────────────
# Etapa 5: Quitar Stopwords
# ──────────────────────────────────────────────
print(SEPARADOR)
print("ETAPA 5: FILTRADO DE STOPWORDS")
print(SEPARADOR)
print()
tokens_sin_stopwords = [token.text for token in doc if not token.is_stop]
texto_limpio = " ".join(tokens_sin_stopwords)
print(f"  Texto original: {TEXTO}")
print()
print(f"  Texto sin stopwords: {texto_limpio}")
print()
print(SEPARADOR)
print("Pipeline completado con éxito.")
print(SEPARADOR)
