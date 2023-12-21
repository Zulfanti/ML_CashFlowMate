from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        features = data['features']

        # Preprocess the input features
        scaled_features = scaler.transform(features)
        scaled_features_dense = csr_matrix.toarray(scaled_features)

        # Make predictions using the loaded model
        predictions = model.predict(scaled_features_dense)

        # Return the predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
