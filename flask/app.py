# api_xtts.py
import os
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import tempfile # Para manejar archivos temporales de forma segura

# --- Configuración ---
# Cambia a False si no tienes una GPU o si PyTorch no está configurado para CUDA
USE_GPU = True
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
# Directorio para guardar temporalmente los archivos de audio subidos y generados
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Inicialización de Flask y TTS ---
app = Flask(__name__)

# Inicializar TTS. Esto puede tardar un poco la primera vez que se descarga el modelo.
# Es mejor inicializarlo una vez al arrancar la aplicación para evitar recargarlo en cada request.
try:
    print(f"Cargando modelo TTS: {MODEL_NAME} (GPU: {USE_GPU})...")
    tts_model = TTS(MODEL_NAME, gpu=USE_GPU)
    print("Modelo TTS cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo TTS: {e}")
    tts_model = None # Manejar el caso donde el modelo no se pueda cargar

@app.route('/tts', methods=['POST'])
def text_to_speech_api():
    """
    Endpoint para generar voz a partir de texto usando clonación de voz.
    Espera datos de formulario (form-data) con:
    - 'text': El texto a convertir en voz.
    - 'language': El código del idioma (ej. 'en', 'es', 'fr').
    - 'speaker_wav': El archivo de audio .wav para clonar la voz.
    """
    if tts_model is None:
        return jsonify({"error": "Modelo TTS no está disponible. Revisa los logs del servidor."}), 500

    if 'text' not in request.form:
        return jsonify({"error": "Falta el parámetro 'text'"}), 400
    if 'language' not in request.form:
        return jsonify({"error": "Falta el parámetro 'language'"}), 400
    if 'speaker_wav' not in request.files:
        return jsonify({"error": "Falta el archivo 'speaker_wav'"}), 400

    text_to_convert = request.form['text']
    language_code = request.form['language']
    speaker_file = request.files['speaker_wav']

    if not speaker_file.filename.lower().endswith('.wav'):
        return jsonify({"error": "El archivo 'speaker_wav' debe ser un archivo .wav"}), 400

    # Guardar el archivo speaker_wav temporalmente
    # Usar tempfile para mayor seguridad y limpieza automática si es necesario,
    # o un nombre único si se prefiere guardarlo explícitamente.
    temp_speaker_path = os.path.join(UPLOAD_FOLDER, f"temp_speaker_{os.urandom(8).hex()}.wav")
    speaker_file.save(temp_speaker_path)

    # Definir la ruta del archivo de salida
    # Usar un nombre único para el archivo de salida para evitar colisiones
    output_filename = f"output_{os.urandom(8).hex()}.wav"
    output_file_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        print(f"Generando audio para el texto: '{text_to_convert[:50]}...' en idioma '{language_code}'")
        # Generar la voz
        tts_model.tts_to_file(
            text=text_to_convert,
            file_path=output_file_path,
            speaker_wav=temp_speaker_path,
            language=language_code
        )
        print(f"Audio generado y guardado en: {output_file_path}")

        # Enviar el archivo de audio generado como respuesta
        # cleanup_stop_signals=False es importante si Flask se ejecuta en un solo hilo
        # y el envío del archivo es bloqueante.
        # Para producción, considera servir archivos estáticos a través de un servidor web como Nginx.
        response = send_file(output_file_path, as_attachment=True, download_name=output_filename, mimetype='audio/wav')
        
        # Opcional: Limpiar los archivos después de enviarlos si no se necesitan más.
        # Es mejor hacerlo en un bloque finally o con un gestor de contexto.
        # @response.call_on_close
        # def cleanup_files_on_close():
        #     try:
        #         if os.path.exists(temp_speaker_path):
        #             os.remove(temp_speaker_path)
        #         if os.path.exists(output_file_path):
        #             os.remove(output_file_path)
        #         print("Archivos temporales limpiados.")
        #     except Exception as e:
        #         print(f"Error limpiando archivos temporales: {e}")
        
        return response

    except Exception as e:
        print(f"Error durante la generación de TTS: {e}")
        # Limpiar en caso de error también
        if os.path.exists(temp_speaker_path):
            os.remove(temp_speaker_path)
        if os.path.exists(output_file_path) and os.path.isfile(output_file_path): # Asegurarse que es un archivo antes de borrar
             os.remove(output_file_path)
        return jsonify({"error": f"Error en la generación de TTS: {str(e)}"}), 500
    finally:
        # Limpieza final de archivos temporales si no se usa call_on_close
        # (call_on_close puede ser más fiable para la limpieza post-envío)
        # Si no se usa call_on_close, el archivo de salida se borrará antes de que el cliente lo descargue completamente.
        # Por eso, para este ejemplo simple, no borraremos el output_file_path aquí.
        # El temp_speaker_path sí se puede borrar.
        if os.path.exists(temp_speaker_path):
            try:
                os.remove(temp_speaker_path)
                print(f"Archivo de referencia de voz temporal limpiado: {temp_speaker_path}")
            except Exception as e_clean:
                print(f"Error limpiando archivo de referencia de voz temporal: {e_clean}")


if __name__ == '__main__':
    # Ejecutar el servidor Flask
    # host='0.0.0.0' para que sea accesible desde fuera del contenedor/máquina local
    # debug=True es útil para desarrollo, pero desactívalo en producción
    app.run(host='0.0.0.0', port=5005, debug=True)
