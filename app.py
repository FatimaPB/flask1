from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

model = joblib.load('modelo_red_neuronal.pkl')  # Asegúrate de tener tu modelo entrenado para precio
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {
            'CS': float(request.form['CS']),
            'Density': float(request.form['Density']),
            'WC': float(request.form['WC']),
            'pH': float(request.form['pH']),
            'Pollen_analysis': request.form['Pollen_analysis'],
            'Purity': float(request.form['Purity']),
        }

        # Convertir a DataFrame. Se recomienda tener el mismo orden de columnas usado en entrenamiento
        data_df = pd.DataFrame([features])
        app.logger.debug(f'Datos recibidos: {data_df}')

        # Asegúrate de codificar correctamente la columna categórica 'Pollen_analysis' si el modelo lo necesita

        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'precio': round(float(prediction[0]), 2)})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
