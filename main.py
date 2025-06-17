# =============================
# CONFIGURACIÓN Y DEPENDENCIAS
# =============================
from fastapi import FastAPI, Request, Form, Response, Cookie
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature
import os
from dotenv import load_dotenv
import json
import time
import threading
import queue
from google import genai

# Carga variables de entorno desde .env
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# =============================
# PROMPT PARA GEMINI
# =============================
PROMT = """
Eres un generador experto de ejercicios de análisis de código en Python dirigidos a estudiantes principiantes. Tu tarea es crear preguntas de opción múltiple sobre código que utiliza únicamente estructuras secuenciales, sin condicionales, sin bucles y sin funciones definidas por el usuario.

OBJETIVO DIDÁCTICO
El objetivo es reforzar la comprensión de la ejecución secuencial de instrucciones: lectura de variables, operaciones, asignaciones, entrada/salida básica, manejo de strings y aritmética fundamental.

REQUISITOS DEL CÓDIGO
El código debe estar escrito en Python válido, usando la última versión estable.

Utilizá nombres de variables en español, con notación camelCase, simples y representativos.

Debe contener solo instrucciones secuenciales:

Asignación de variables

Cálculos matemáticos (suma, resta, multiplicación, división, módulo, etc.)

Operaciones con strings (concatenación, slicing, longitud, repetición, etc.)

Uso básico de input() y print()

Conversión de tipos (int(), float(), str())

No uses if, for, while, def, return, listas, tuplas, diccionarios, sets ni excepciones.

El código debe tener entre 4 y 8 líneas secuenciales con un orden lógico claro.

Si usás input(), indicá claramente en el enunciado los valores a ingresar y asegurate de que afecten el resultado final.

El código debe ser autocontenible, ejecutable y tener una única salida posible.

VARIEDAD Y ESTILO DE PREGUNTAS
- Selecciona aleatoriamente uno de los siguientes tipos:

- "¿Qué salida tendrá el siguiente código?"

- "¿Qué salida tendrá el siguiente código si se ingresan los siguientes valores?"
(si elegís esta opción, el enunciado debe indicar claramente los valores a ingresar, y el código debe usar input() adecuadamente)
(asegúrate de que los valores de input() sean únicos (valores enteros, flotantes o cadenas), puedes basarte en fechas importantes)

- "¿Qué valor tendrá la variable X al finalizar la ejecución del siguiente código?"

Formato de salida (único objeto JSON):

{
  "Pregunta": "Texto de la pregunta clara y concisa.",
  "Codigo": "Fragmento de código Python válido, autocontenible y bien formateado.",
  "Respuestas": ["Respuesta A", "Respuesta B", "Respuesta C", "Respuesta D"],
  "Respuesta correcta": "Respuesta correcta exactamente igual a una de las opciones",
  "Explicacion": "Explicación breve y general del razonamiento que lleva al resultado correcto, sin hacer referencia a letras o posiciones de respuesta."
}

IMPORTANTE:
No incluir condicionales (if, else, elif), bucles (for, while), funciones definidas (def), ni estructuras de datos.

No incluir comentarios, explicaciones externas, ni formato Markdown.

Responder únicamente con el objeto JSON, sin texto adicional.
"""

# =============================
# CLIENTE GENAI
# =============================
client = genai.Client(api_key=GENAI_API_KEY)

# =============================
# CACHE DE PREGUNTAS (COLA)
# =============================
CACHE_SIZE = 200  # Máximo de preguntas en cache
CACHE_MIN = 100   # Umbral mínimo para reponer el cache
pregunta_cache = queue.Queue(maxsize=CACHE_SIZE)

# =============================
# GENERACIÓN Y OBTENCIÓN DE PREGUNTAS
# =============================
def generar_pregunta():
    """
    Llama a Gemini para generar una pregunta nueva.
    Limpia el texto y lo convierte a un diccionario Python.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=PROMT
    )
    try:
        text = response.text.strip()
        # Limpia el texto de bloques de código Markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        pregunta_json = json.loads(text)
        # Validación de la estructura del JSON
        respuestas = pregunta_json.get("Respuestas")
        if isinstance(respuestas, str):
            respuestas = [r.strip() for r in respuestas.split(",")]
        elif not isinstance(respuestas, list):
            respuestas = []
        pregunta = {
            "pregunta": pregunta_json.get("Pregunta"),
            "codigo": pregunta_json.get("Codigo"),
            "respuestas": respuestas,
            "respuesta_correcta": pregunta_json.get("Respuesta correcta"),
            "explicacion": pregunta_json.get("Explicacion", "")
        }
        return pregunta
    except Exception as e:
        return {"error": "No se pudo extraer el JSON", "detalle": str(e), "texto": response.text}

# =============================
# HILO DE PRECARGA DE PREGUNTAS
# =============================
def precargar_preguntas():
    """
    Hilo en segundo plano que mantiene el cache de preguntas lleno.
    Solo consulta la API si el cache baja del umbral.
    """
    while True:
        if pregunta_cache.qsize() < CACHE_MIN:
            try:
                pregunta = generar_pregunta()
                # Solo la guarda si es válida
                if isinstance(pregunta, dict) and 'pregunta' in pregunta and 'codigo' in pregunta:
                    pregunta_cache.put(pregunta)
                time.sleep(1)  # Espera un segundo antes de volver a intentar
            except Exception as e:
                # Si es un error de cuota, espera más tiempo
                if "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(35)
                else:
                    time.sleep(5)
        else:
            time.sleep(2)  # Espera antes de volver a chequear

# Inicia el hilo de precarga al arrancar la app
threading.Thread(target=precargar_preguntas, daemon=True).start()

def obtener_pregunta_cache():
    """
    Obtiene una pregunta del cache (espera hasta 10s).
    Si el cache está vacío, genera una pregunta en caliente.
    """
    try:
        return pregunta_cache.get(timeout=10)
    except Exception:
        pregunta = generar_pregunta()
        if "error" in pregunta and "RESOURCE_EXHAUSTED" in pregunta.get("detalle", ""):
            return {
                "pregunta": "¡Límite de uso alcanzado!",
                "codigo": "",
                "respuestas": [],
                "respuesta_correcta": "",
                "explicacion": "Se ha superado el límite de uso de la API. Por favor, espera un minuto y vuelve a intentarlo."
            }
        return pregunta

# =============================
# FASTAPI APP Y RUTAS
# =============================

# Inicialización de la app y sistema de plantillas
app = FastAPI()
templates_path = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_path)

# Configuración de la clave secreta y serializador para cookies firmadas
SECRET_KEY = os.getenv("SESSION_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SESSION_SECRET_KEY no está configurada. "
        "Establezca un valor seguro en su entorno o archivo .env."
    )
SESSION_COOKIE = "quiz_session"
serializer = URLSafeSerializer(SECRET_KEY)

def get_session(request: Request):
    """
    Recupera y valida la sesión del usuario desde la cookie.
    Si no existe o la firma es inválida, devuelve un dict vacío.
    """
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return {}
    try:
        return serializer.loads(cookie)
    except BadSignature:
        return {}

def set_session(response: Response, session_data: dict):
    """
    Serializa y firma los datos de sesión, y los guarda en la cookie de la respuesta.
    """
    cookie_value = serializer.dumps(session_data)
    response.set_cookie(SESSION_COOKIE, cookie_value, httponly=True, max_age=60*60*2)

def clear_session(response: Response):
    """
    Elimina la cookie de sesión.
    """
    response.delete_cookie(SESSION_COOKIE)

@app.get('/', name="inicio")
def inicio(request: Request):
    """
    Ruta de inicio: muestra la presentación y botón para comenzar el quiz.
    Limpia cualquier sesión previa.
    """
    response = templates.TemplateResponse('inicio.html', {'request': request})
    clear_session(response)
    return response

@app.get("/quiz", name="quiz")
async def quiz_get(request: Request):
    """
    Muestra la pregunta actual.
    Si la sesión no existe o está incompleta, la inicializa.
    """
    session = get_session(request)
    if not all(k in session for k in ['puntaje', 'total', 'inicio', 'pregunta_actual', 'errores']) or session == {}:
        session = {
            'puntaje': 0,
            'total': 0,
            'inicio': int(time.time()),
            'pregunta_actual': obtener_pregunta_cache(),
            'errores': []
        }
    pregunta = session['pregunta_actual']
    num_pregunta = session.get('total', 0) + 1
    response = templates.TemplateResponse(
        'quiz.html',
        {'request': request, 'pregunta': pregunta, 'num_pregunta': num_pregunta}
    )
    set_session(response, session)
    return response

@app.post('/quiz')
async def quiz_post(request: Request, respuesta: str = Form(...)):
    """
    Procesa la respuesta del usuario y muestra la siguiente pregunta o el resultado.
    Actualiza el puntaje y los errores en la sesión.
    Si se llega a 10 preguntas, redirige a la página de resultados.
    """
    session = get_session(request)
    if not all(k in session for k in ['puntaje', 'total', 'inicio', 'pregunta_actual', 'errores']):
        # Si no hay sesión, redirige al inicio
        return RedirectResponse(url='/', status_code=303)

    seleccion = respuesta
    correcta = session['pregunta_actual']['respuesta_correcta']
    explicacion = session['pregunta_actual']['explicacion']
    session['total'] += 1
    if seleccion and seleccion.strip() == correcta.strip():
        session['puntaje'] += 1
    else:
        errores = session.get('errores', [])
        errores.append({
            'pregunta': session['pregunta_actual']['pregunta'],
            'codigo': session['pregunta_actual']['codigo'],
            'respuestas': session['pregunta_actual']['respuestas'],
            'respuesta_correcta': correcta,
            'explicacion': explicacion,
            'respuesta_usuario': seleccion
        })
        session['errores'] = errores

    if session['total'] >= 10:
        # Si ya respondió 10 preguntas, calcula el tiempo y redirige a resultados
        tiempo = int(time.time() - session['inicio'])
        puntaje = session['puntaje']
        errores = session.get('errores', [])
        response = RedirectResponse(
            url=f'/resultado?correctas={puntaje}&tiempo={tiempo}',
            status_code=303
        )
        clear_session(response)
        # Guarda los errores en una cookie temporal para mostrar en resultado
        response.set_cookie("quiz_errores", serializer.dumps(errores), max_age=60*5)
        return response

    # Si no ha terminado, obtiene una nueva pregunta y actualiza la sesión
    session['pregunta_actual'] = obtener_pregunta_cache()
    response = RedirectResponse(url='/quiz', status_code=303)
    set_session(response, session)
    return response

@app.get('/resultado')
def resultado(request: Request, correctas: int = 0, tiempo: int = 0, quiz_errores: str = Cookie(default=None)):
    """
    Ruta para mostrar el resultado final.
    Recupera los errores desde la cookie temporal y los muestra junto al puntaje y tiempo.
    """
    errores = []
    if quiz_errores:
        try:
            errores = serializer.loads(quiz_errores)
        except BadSignature:
            errores = []
    response = templates.TemplateResponse(
        'resultado.html',
        {'request': request, 'correctas': correctas, 'tiempo': tiempo, 'errores': errores}
    )
    response.delete_cookie("quiz_errores")
    return response

@app.get('/error')
def error(request: Request, detalle: str = '', texto: str = ''):
    """
    Ruta para mostrar errores personalizados.
    """
    return templates.TemplateResponse(
        'error.html',
        {'request': request, 'detalle': detalle, 'texto': texto},
        status_code=500
    )