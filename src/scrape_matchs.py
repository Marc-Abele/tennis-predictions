from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
from os import path
import pandas as pd
from src.config import (
    URL, 
    PREFIX_MATCH_ID,
    URL_MATCH_PREFIX,
    URL_MATCH_SUFFIX,
    MAPPING_SURFACE,
    MAPPING_LOCATION_SERIES,
    PATH_DF_PRED,
    MAPPING_ROUNDS_INF_1000,
    MAPPING_ROUNDS_SUP_1000)

def get_ids_match():
    """Scrape webpage and return a list of match ids

    Returns:
        ids_match: list
    """
    # récupération du contenu de la page
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(URL)
    except Exception as e:
        print(f"Error: Can't open url: {URL} {e}")
        driver.quit()
        exit()

    page_content = driver.page_source
    try:
        soup = BeautifulSoup(page_content, "html.parser")
    except Exception as e:
        print(f"Error: Can't create soup object: {e}")
        driver.quit()
        exit()
    driver.quit()
    div=soup.find_all('div', {'title':'Cliquez pour les détails du match!'})
    
    # récupération des id de chaque match, servant à construire l'url
    ids_match = []
    for match in div:
        ids_match.append(match.get('id'))
    return ids_match
    
def parse_match_page(url):
    """Scrape a tennis match webpage and return html content

    Args:
        url (str): Link to the match webpage

    Returns:
        soup_match: BeautifoulSoup html content
    """
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
    except Exception as e:
        print(f"Error: Can't open url: {url} {str(e)}")
        driver.quit()
        exit()
    
    # Attends quelques secondes pour permettre le chargement du contenu par JavaScript
    driver.implicitly_wait(5)    
    page_content = driver.page_source
    try:
        soup_match = BeautifulSoup(page_content, "html.parser")
    except Exception as e:
        print(f"Error: Can't create soup object: {str(e)}")
        driver.quit()
        exit()
    driver.quit()
    return soup_match

pattern = re.compile(r'ATP - SIMPLES|WTA - SIMPLES')

def check_atp_wta(text: str):
    return bool(pattern.search(text))

def extract_rank(text: str):
    rank = re.search(r'\d+', text).group()
    return int(rank)

def check_match_status(content: str):
    try:
        status = content.find_all("span", {"class":"fixedHeaderDuel__detailStatus"})[0].text
    except AttributeError:
        status = ""
    return status

def parse_title(title: str):
    """Parse a string to find Serie, city, surface of the match and round

    Args:
        title (str): Description given when parsing the match page

    Returns:
        serie, city, surface, round: str
    """
    try:
        serie = title.split(" - ")[0]
        city = re.search(r':\s(.*?)(?:\s*\()', title).group(1)
        surface = title.split(",")[-1].strip()
        round = re.search(r'\s-\s(.*?-\s(.*?))$', title).group(2)
    except AttributeError:
        print(f"Impossible to parse title: {title}")
    return serie, city, surface, round

def mapping_round(series_value: str, round_value: str):
    """Apply a mapping on rounds to translate from fr. to en. depending of series
    
    Args:
        series_value (str): Rank tournament (eg: ATP250, ATP500)
        round_value (str): Round in the tournament

    Returns:
        round: str
    """
    if series_value in ["ATP250", "ATP500"]:
        return MAPPING_ROUNDS_INF_1000.get(round_value, round_value)
    else:
        return MAPPING_ROUNDS_SUP_1000.get(round_value, round_value)

def prepare_dataset_for_pred(df: pd.DataFrame):
    df["Surface"] = df["Surface"].map(MAPPING_SURFACE)
    df["Series"] = df["Series"].map(MAPPING_LOCATION_SERIES)
    df["Round"] = df.apply(lambda row: mapping_round(row["Series"], row["Round"]))
    return df

def save_df_for_pred(df: pd.DataFrame):
    day = pd.Timestamp.now(tz="CET").strftime("%-%m-%d")
    path_file = f"{PATH_DF_PRED}{day}.csv"
    if path.exists(path_file):
        df_tmp = pd.read_csv(path_file)
    else:
        df_tmp = pd.DataFrame()
    df = pd.concat([df_tmp, df])
    df.drop_duplicates(keep="last", inplace=True)
    df.to_csv(path_file)
    
def get_datas_per_match(ids_match):
    player1 = []
    player2 = []
    rank1 = []
    rank2 = []
    date = []
    cote1 = []
    cote2 = []
    city = []
    surface = []
    serie = []
    round = []
    court = [] # outdoor/indoor

    for id in ids_match:
        id_clean = id.split(PREFIX_MATCH_ID)[-1]
        url_match = f"{URL_MATCH_PREFIX}{id_clean}{URL_MATCH_SUFFIX}"
        print(url_match)
        content = parse_match_page(url_match)
        status = check_match_status(content)
        if status == "Annulé":
            continue
        else:
            title = content.find_all("span", {"class" : "tournamentHeader__country"})[0].text
            if check_atp_wta(title):
                try:
                    cotes = content.find_all("span", {"class":"oddsValueInner"})
                    cote1.append(float(cotes[-2].text))
                    cote2.append(float(cotes[-1].text))
                    serie0, city0, surface0, round0 = parse_title(title)
                    city.append(city0)
                    surface.append(surface0)
                    serie.append(serie0)
                    round.append(round0)
                    court.append("Outdoor")
                
                    players = content.find_all("div", {"class" : "participant__participantNameWrapper"})
                    player1.append(players[0].text)
                    player2.append(players[-1].text)
                    
                    ranks = content.find_all("div", {"class":"participant__participantRank"})
                    rank1.append(extract_rank(ranks[0].text))
                    rank2.append(extract_rank(ranks[-1].text))
                    
                    date_start = content.find("div", {"class":"duelParticipant__startTime"}).text
                    date_start = pd.to_datetime(date_start, format='%d.%m.%Y %H:%M').normalize()
                    date.append(date_start)
                    print(f"Match {url_match} OK.")
                except IndexError:
                    print(f"Not enough data for this match : {url_match}")
                    continue
            else:
                continue
    df_tmp = pd.DataFrame({
        "Player1": player1,
        "Player2": player2,
        "Date": date,
        "Location": city,
        "Tournament": city,
        "Series": serie,
        "Court": court,
        "Surface": surface,
        "Round": round,
        "Rank1": rank1,
        "Rank2": rank2,
        "Cote1": cote1,
        "Cote2": cote2,
        })

    return df_tmp

def main():
    ids = get_ids_match()
    df = get_datas_per_match(ids)
    df = prepare_dataset_for_pred(df)
    save_df_for_pred(df)
    print(f"{len(df)} matchs found.")
    
    
if __name__ == '__main__':
    main()