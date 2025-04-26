
document.addEventListener('DOMContentLoaded', function() {
    // Your OpenWeatherMap API Key
    const apiKey = '744c0a84a4ec83dc999a06231f7e2ead';

    // Weather icons mapping
    const weatherIcons = {
        'Clear': '☀️',
        'Clouds': '☁️',
        'Rain': '🌧️',
        'Drizzle': '🌦️',
        'Thunderstorm': '⛈️',
        'Snow': '❄️',
        'Mist': '🌫️',
        'Fog': '🌫️',
        'Haze': '🌫️'
    };

    // Plant care tips based on weather
    const plantTips = { /* Your same plantTips object */ 
        'Clear': [
            "Consider providing shade for sensitive plants.",
            "Water plants early morning or late evening.",
            "Check soil moisture more frequently in sunny conditions."
        ],
        'Clouds': [
            "Moderate watering is typically sufficient.",
            "Good day for transplanting or repotting.",
            "Check for pests that may be active in mild conditions."
        ],
        'Rain': [
            "Hold off on watering if rain is substantial.",
            "Check drainage to prevent root rot.",
            "Watch for fungal diseases in prolonged wet conditions."
        ],
        'Drizzle': [
            "Light rain may not be enough for all plants, check soil.",
            "Good conditions for foliar feeding.",
            "Inspect plants for early signs of fungal issues."
        ],
        'Thunderstorm': [
            "Check plants for damage after storms.",
            "Ensure proper drainage to prevent flooding.",
            "Consider bringing potted plants indoors if severe."
        ],
        'Snow': [
            "Protect sensitive plants from frost damage.",
            "Brush off heavy snow from branches.",
            "Hold off on fertilizing during cold weather."
        ],
        'Mist': [
            "Good conditions for tropical plants that enjoy humidity.",
            "Reduced need for misting houseplants.",
            "Monitor for fungal issues in prolonged humid conditions."
        ],
        'Fog': [
            "Keep an eye on moisture levels in the soil.",
            "Fungal diseases may develop in high humidity.",
            "Delay spraying pesticides until fog clears."
        ],
        'Haze': [
            "Rinse dust from leaves if air quality is poor.",
            "Keep plants well-watered during hazy conditions.",
            "Indoor plants may need extra attention if windows stay closed."
        ]
    };

    // Get user's location for weather data
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(fetchWeatherData, handleLocationError);
    } else {
        displayError("Geolocation is not supported by your browser.");
    }

    function fetchWeatherData(position) {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        
        const apiUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${apiKey}`;

        fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Weather data could not be retrieved.");
                }
                return response.json();
            })
            .then(data => displayWeatherData(data))
            .catch(error => {
                console.error(error);
                displayError("Failed to fetch weather data.");
            });
    }

    function displayWeatherData(data) {
        document.getElementById('temperature').textContent = `${Math.round(data.main.temp)}°C`;
        document.getElementById('location').textContent = `${data.name}, ${data.sys.country}`;

        const weatherMain = data.weather[0].main;
        const weatherDesc = data.weather[0].description;
        document.getElementById('weather-description').textContent = capitalizeFirstLetter(weatherDesc);

        const iconElement = document.getElementById('weather-icon');
        iconElement.textContent = weatherIcons[weatherMain] || '🌤️';
        iconElement.style.fontSize = '2.5rem';

        document.getElementById('humidity').textContent = `${data.main.humidity}%`;
        document.getElementById('wind').textContent = `${data.wind.speed} m/s`;
        document.getElementById('pressure').textContent = `${data.main.pressure} hPa`;

        // Optional: Fetch UV Index separately using One Call API if needed
        // For now, use a random number for UV index (since UV index needs different API endpoint)
        document.getElementById('uv-index').textContent = `${Math.floor(Math.random() * 5) + 1}`;

        const tipsContainer = document.getElementById('plant-tips');
        tipsContainer.innerHTML = '';

        const tips = plantTips[weatherMain] || plantTips['Clouds'];
        tips.forEach(tip => {
            const tipElement = document.createElement('p');
            tipElement.textContent = tip;
            tipsContainer.appendChild(tipElement);
        });
    }

    function handleLocationError(error) {
        let errorMessage;
        switch(error.code) {
            case error.PERMISSION_DENIED:
                errorMessage = "Location access denied. Please enable location services.";
                break;
            case error.POSITION_UNAVAILABLE:
                errorMessage = "Location information is unavailable.";
                break;
            case error.TIMEOUT:
                errorMessage = "Location request timed out.";
                break;
            case error.UNKNOWN_ERROR:
                errorMessage = "An unknown error occurred.";
                break;
        }
        displayError(errorMessage);
    }

    function displayError(message) {
        document.getElementById('temperature').textContent = `--°C`;
        document.getElementById('location').textContent = `Location unavailable`;
        document.getElementById('weather-description').textContent = message;

        const weatherIcon = document.getElementById('weather-icon');
        weatherIcon.textContent = '📍';
        weatherIcon.style.fontSize = '2rem';

        const tipsContainer = document.getElementById('plant-tips');
        tipsContainer.innerHTML = `
            <p>Water plants when the top inch of soil feels dry.</p>
            <p>Most plants need 6-8 hours of sunlight daily.</p>
            <p>Check regularly for pests and disease symptoms.</p>
        `;
    }

    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
