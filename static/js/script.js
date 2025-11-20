document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Prediction Class: ' + data.prediction;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});