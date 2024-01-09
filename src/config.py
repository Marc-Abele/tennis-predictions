trained_algo = ""
PREFIX_MATCH_ID = "g_2_"
URL = "https://www.flashscore.fr/tennis/"
URL_MATCH_PREFIX = "https://www.flashscore.fr/match/"
URL_MATCH_SUFFIX = "/#/resume-du-match"

PATH_DF_PRED = "data/data_pred/"

test = 1
test2 = 2

MAPPING_SURFACE = {
    "dur": "Hard",
    "gazon": "Grass",
    "terre battue": "Clay"
}

MAPPING_LOCATION_SERIES = {
    "Halle": 'ATP500',
    'Queens Club': 'ATP250',
    'London': 'Grand Slam',
    'Bastad': 'ATP250',
    'Gstaad': 'ATP250',
    'Newport': 'ATP250',
    'Amersfoort': 'International',
    'Stuttgart': 'ATP250',
    'Umag': 'ATP250',
    'Kitzbuhel': 'ATP250',
    'Los Angeles': 'International',
    'Sopot': 'International',
    'Toronto': 'Masters 1000',
    'Cincinnati': 'Masters 1000',
    'Indianapolis': 'International',
    'Washington': 'ATP500',
    'Long Island': 'International',
    'New York': 'Grand Slam',
    'Bucharest': 'ATP250',
    'Salvador': 'International',
    'Tashkent': 'International',
    'Hong Kong': 'International',
    'Palermo': 'International',
    'Moscow': 'ATP250',
    'Tokyo': 'ATP500',
    'Lyon': 'International',
    'Vienna': 'ATP500',
    'Madrid': 'Masters 1000',
    'Basel': 'ATP500',
    'St. Petersburg': 'ATP250',
    'Stockholm': 'ATP250',
    'Paris': 'Grand Slam',
    'Shanghai': 'Masters 1000',
    'Adelaide': 'International',
    'Chennai': 'ATP250',
    'Doha': 'ATP250',
    'Auckland': 'ATP250',
    'Sydney': 'ATP250',
    'Melbourne': 'Grand Slam',
    'Milan': 'International',
    'Marseille': 'ATP250',
    'San Jose': 'International',
    'Vina del Mar': 'International',
    'Buenos Aires': 'ATP250',
    'Memphis': 'International Gold',
    'Rotterdam': 'ATP500',
    'Acapulco': 'ATP500',
    'Copenhagen': 'International',
    'Dubai ': 'ATP500',
    'Delray Beach': 'ATP250',
    'Scottsdale': 'International',
    'Indian Wells': 'Masters 1000',
    'Miami': 'Masters 1000',
    'Casablanca': 'ATP250',
    'Estoril ': 'International',
    'Monte Carlo': 'Masters 1000',
    'Barcelona': 'ATP500',
    'Houston': 'ATP250',
    'Munich': 'ATP250',
    'Valencia': 'ATP500',
    'Rome': 'Masters 1000',
    'Hamburg': 'ATP500',
    'St. Polten': 'International',
    'Nottingham': 'International',
    "'s-Hertogenbosch": 'ATP250',
    'Montreal': 'Masters 1000',
    'Costa Do Sauipe': 'International',
    'Bangkok': 'International',
    'Metz': 'ATP250',
    'Vienna ': 'International Gold',
    'Beijing': 'ATP500',
    'New Haven': 'International Gold',
    'Ho Chi Min City': 'International',
    'Zagreb': 'ATP250',
    'Las Vegas': 'International',
    'Portschach': 'International',
    'Mumbai': 'International',
    'Warsaw': 'International',
    'Brisbane': 'ATP250',
    'Johannesburg ': 'ATP250',
    'Belgrade': 'ATP250',
    'Eastbourne': 'ATP250',
    'Kuala Lumpur': 'ATP250',
    'Santiago': 'ATP250',
    'Nice': 'ATP250',
    'Atlanta': 'ATP250',
    'Montpellier': 'ATP250',
    'Winston-Salem': 'ATP250',
    'Sao Paulo': 'ATP250',
    'Oeiras': 'ATP250',
    'Dusseldorf': 'ATP250',
    'Bogota': 'ATP250',
    'Rio de Janeiro': 'ATP500',
    'Shenzhen ': 'ATP250',
    'Quito': 'ATP250',
    'Estoril': 'ATP250',
    'Istanbul': 'ATP250',
    'Geneva': 'ATP250',
    'Sofia': 'ATP250',
    'Marrakech': 'ATP250',
    'Los Cabos': 'ATP250',
    'Chengdu': 'ATP250',
    'Antwerp': 'ATP250',
    'Budapest': 'ATP250',
    'Antalya': 'ATP250',
    'Pune': 'ATP250',
    'Cordoba': 'ATP250',
    'Zhuhai': 'ATP250',
    'Cologne': 'ATP250',
    'Sardinia': 'ATP250',
    'Nur-Sultan': 'ATP250',
    'Singapore': 'ATP250',
    'Cagliari': 'ATP250',
    'Marbella': 'ATP250',
    'Parma': 'ATP250',
    'Mallorca': 'ATP250',
    'San Diego': 'ATP250',
    'Turin': 'Masters Cup',
    'Dallas': 'ATP250',
    'Seoul': 'ATP250',
    'Tel Aviv': 'ATP250',
    'Florence': 'ATP250',
    'Gijon': 'ATP250',
    'Napoli': 'ATP250',
    'Banja Luka': 'ATP250'
    }

MAPPING_ROUNDS_INF_1000 = {
    "1/32 DE FINALE": "1st Round",
    "1/16 DE FINALE": "2nd Round",
    "1/8 DE FINALE": "3rd Round",
    "QUARTS DE FINALE": "Quarterfinals",
    "DEMI-FINALES": "Semifinals",
    "FINALE": "The Final",
    "ROUND ROBIN": "Round Robin"
    }

MAPPING_ROUNDS_SUP_1000 = {
    "1/64 DE FINALE": "1st Round",
    "1/32 DE FINALE": "2nd Round",
    "1/16 DE FINALE": "3rd Round",
    "1/8 DE FINALE": "4th Round",
    "QUARTS DE FINALE": "Quarterfinals",
    "DEMI-FINALES": "Semifinals",
    "FINALE": "The Final",
    "ROUND ROBIN": "Round Robin"

}