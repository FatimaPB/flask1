from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo, encoder y scaler
model = joblib.load('modelo_random_forest.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo, encoder y scaler cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener y limpiar entrada del formulario
        pollen_input = request.form['Pollen_analysis'].strip()
        pollen_encoded = ordinal_encoder.transform([[pollen_input]])[0][0]

        features = {
            'Purity': float(request.form['Purity']),
            'pH': float(request.form['pH']),
            'Density': float(request.form['Density']),
            'CS': float(request.form['CS']),
            'WC': float(request.form['WC']),
            'Pollen_analysis': pollen_encoded,
        }

        # Convertir a DataFrame
        data_df = pd.DataFrame([features])

        # ✅ Reordenar columnas como en el entrenamiento
        data_df = data_df[['Purity', 'pH', 'Density', 'CS', 'WC', 'Pollen_analysis']]
        app.logger.debug(f'Datos recibidos ordenados: {data_df}')

        # Escalar los datos con el mismo scaler que se usó en entrenamiento
        data_scaled = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {data_scaled}')

        # Predicción
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')

        return jsonify({'precio': round(float(prediction[0]), 2)})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
