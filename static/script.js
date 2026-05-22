document.addEventListener('DOMContentLoaded', () => {

    const form = document.getElementById('attritionForm');

    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', async (event) => {

        event.preventDefault();

        const formData = new FormData(form);

        const data = {};

        formData.forEach((value, key) => {

            data[key] = value;

        });

        try {

            const response = await fetch('/predict', {

                method: 'POST',

                headers: {
                    'Content-Type': 'application/json',
                },

                body: JSON.stringify(data)

            });

            const result = await response.json();

            if (response.ok) {

                resultDiv.innerHTML = `
                    <strong>${result.prediction}</strong>
                    <br>
                    Probability: ${(result.probability * 100).toFixed(2)}%
                `;

            } else {

                resultDiv.textContent =
                    `Error: ${result.error}`;

            }

        } catch (error) {

            resultDiv.textContent =
                `Error: ${error.message}`;

        }

    });

});