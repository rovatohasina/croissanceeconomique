import wbdata
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Prévision du PIB", layout="wide", initial_sidebar_state="expanded")
st.markdown( 
    """
    <style>
        body {
            background-color: #0F4C60;
            color: white;
        }
        .stApp {
            background-color: #D3D3D3;
        }
         color: #FCA311;
        # }
        .stTitle {
            color: white;
        }
        .st-emotion-cache-t74pzu {
            width: calc(50% - 1rem);
            flex: 1 1 calc(50% - 1rem);
            background-color: white;
            padding: 16px;
            border-radius: 16px;
}
    </style>
    """,
    unsafe_allow_html=True
)
# Indicateurs économiques
indicators = {
    "NY.GDP.MKTP.KD": "PIB",
    "NE.EXP.GNFS.ZS": "Exportations",
    "NE.IMP.GNFS.ZS": "Importations", 
    "NY.GDP.PCAP.KD": "PIB par habitant",
    "SL.TLF.CACT.ZS": "Participation au marché du travail",
    "PA.NUS.FCRF": "Taux de change officiel",
    "GC.XPN.TOTL.GD.ZS": "Dépenses publiques",
    "SL.UEM.TOTL.ZS": "Chômage",
    "FP.CPI.TOTL.ZG": "Inflation",
    "BX.KLT.DINV.WD.GD.ZS": "Investissements directs étrangers entrées nettes",
    "BM.KLT.DINV.WD.GD.ZS": "Investissements directs étrangers sortie nettes"
}
indicators_ventilation_depenses = {
    "GC.XPN.INTP.ZS": "Paiements d'intérêts",
    "MS.MIL.XPND.ZS": "Dépenses militaires",
    "SH.XPD.GHED.GE.ZS": "Dépenses de santé",
    "SE.XPD.TOTL.GB.ZS": "Dépenses publiques d'éducation",
    "SE.XPD.CTOT.ZS": "Dépenses courantes d'éducation",
}
indicators_depenses = {
    "GC.XPN.TOTL.GD.ZS": "Dépenses publiques",
    "GC.XPN.COMP.ZS": "Rémunération des employés",
    "NE.DAB.TOTL.ZS": "Dépenses nationales brutes",
    "NE.CON.TOTL.ZS": "Dépenses de consommation finale"
}
indicators_recettes = {
    "GC.TAX.TOTL.GD.ZS": "Recettes fiscales",
    "GC.REV.XGRT.GD.ZS": "Recettes, hors subventions",
    "GC.TAX.YPKG.ZS": "Impôts sur le revenu"
}

# Récupération des données
df = wbdata.get_dataframe(indicators, country="MDG")
df.reset_index(inplace=True)
df.rename(columns={'date': 'Year'}, inplace=True)
df = df.dropna(subset=['PIB','Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes'], how='all')
df['Year'] = df['Year'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)

# completer les valeurs Nan
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
#reverser les donnee
df = df.iloc[::-1]
df_depenses = wbdata.get_dataframe(indicators_depenses, country="MDG")
df_depenses.reset_index(inplace=True)
df_depenses.rename(columns={'date': 'Year'}, inplace=True)
df_depenses = df_depenses.dropna(subset=[ "Dépenses publiques","Rémunération des employés","Dépenses nationales brutes","Dépenses de consommation finale"], how='all')
df_depenses['Year'] = df_depenses['Year'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)

df_recettes = wbdata.get_dataframe(indicators_recettes, country="MDG")
df_recettes.reset_index(inplace=True)
df_recettes.rename(columns={'date': 'Year'}, inplace=True)
df_recettes = df_recettes.dropna(subset=["Recettes fiscales","Recettes, hors subventions","Impôts sur le revenu"], how='all')
df_recettes['Year'] = df_recettes['Year'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)

ventilation = wbdata.get_dataframe(indicators_ventilation_depenses, country="MDG") 
ventilation.reset_index(inplace=True)
ventilation.rename(columns={'date': 'Year'}, inplace=True)
ventilation = ventilation.dropna(subset=["Paiements d'intérêts","Dépenses militaires","Dépenses de santé","Dépenses publiques d'éducation","Dépenses courantes d'éducation",], how='all')
ventilation['Year'] = ventilation['Year'].astype(int)
# Sidebar
st.sidebar.header("Filtrer par année")
min_year, max_year = df['Year'].min(), df['Year'].max()
selected_years = st.sidebar.slider("Sélectionner une plage d'années", min_year, max_year, (min_year, max_year), 1)
filtered_data = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
filtered_data.dropna(inplace=True)
# Titre
st.title("Analyse de la croissance économique")




# Mise en page avec les valeurs en pourcentage
year_current = selected_years[1]
year_previous = year_current - 1
st.write("")

# Fonction pour l'analyse
def analyser_evolution(pourcentage, seuil=1.0):
    if pourcentage > seuil:
        return "En augmentation"
    elif pourcentage < -seuil:
        return "En légère baisse"
    else:
        return "Stable"

def analyser_balance(export, import_):
    if import_ > export:
        return "Déficit commercial"
    elif export > import_:
        return "Excédent commercial"
    else:
        return "Équilibre"
      
pib_current = filtered_data['PIB'].dropna().iloc[-1]
pib_previous = filtered_data['PIB'].dropna().iloc[-2]
difference_pib = pib_current - pib_previous
pib_percentage = ((difference_pib) / pib_previous)*100

pib_hab_current = filtered_data['PIB par habitant'].dropna().iloc[-1]
pib_hab_previous = filtered_data['PIB par habitant'].dropna().iloc[-2]
difference_pib_hab = pib_hab_current - pib_hab_previous
pib_hab_percentage = ((difference_pib_hab) / pib_hab_previous)*100

export_current = filtered_data['Exportations'].dropna().iloc[-1]
export_previous = filtered_data['Exportations'].dropna().iloc[-2]
difference_export = export_current - export_previous
export_percentage = ((difference_export) / export_previous)*100

import_current = filtered_data['Importations'].dropna().iloc[-1]
import_previous = filtered_data['Importations'].dropna().iloc[-2]
difference_import = import_current - import_previous
import_percentage = ((difference_import) / import_previous)*100

infllation_current = filtered_data['Inflation'].dropna().iloc[-1]
infllation_previous = filtered_data['Inflation'].dropna().iloc[-2]
difference_infllation = infllation_current - infllation_previous
infllation_percentage = ((difference_infllation) / infllation_previous)*100

change_current = filtered_data['Taux de change officiel'].dropna().iloc[-1]
change_previous = filtered_data['Taux de change officiel'].dropna().iloc[-2]
difference_change = change_current - change_previous
change_percentage = ((difference_change) / change_previous)*100

chomage_current = filtered_data['Chômage'].dropna().iloc[-1]
chomage_previous = filtered_data['Chômage'].dropna().iloc[-2]
difference_chomage = chomage_current - chomage_previous
chomage_percentage = ((difference_chomage) / chomage_previous)*100

participation_current = filtered_data['Participation au marché du travail'].dropna().iloc[-1]
participation_previous = filtered_data['Participation au marché du travail'].dropna().iloc[-2]
difference_participation = participation_current - participation_previous
participation_percentage = ((difference_participation) / participation_previous)*100

sortie_current = filtered_data['Investissements directs étrangers sortie nettes'].dropna().iloc[-1]
sortie_previous = filtered_data['Investissements directs étrangers sortie nettes'].dropna().iloc[-2]
difference_sortie = sortie_current - sortie_previous
sortie_percentage = ((difference_sortie) / sortie_previous)*100

entre_current = filtered_data['Investissements directs étrangers entrées nettes'].dropna().iloc[-1]
entre_previous = filtered_data['Investissements directs étrangers entrées nettes'].dropna().iloc[-2]
difference_entre = entre_current - entre_previous
entre_percentage = ((difference_entre) / entre_previous)*100

# Analyse balance commerciale
balance_commerciale = analyser_balance(export_current, import_current)
# balance_commerciale = "Déficit commercial" if import_current > export_current else "Excédent commercial"

df_indicateurs = pd.DataFrame({
    "Indicateur": ["PIB", "PIB par habitant", "Exportations", "Importations","Inflation","Taux de change officiel","Chômage","Participation au marché du travail","Investissements directs étrangers sortie nettes","Investissements directs étrangers entrées nettes"],
    f"Valeur en {year_current}": [
        f"{pib_current:,.2f}",
        f"{pib_hab_current:,.2f}",
        f"{export_current:,.2f}",
        f"{import_current:,.2f}",
        f"{infllation_current:,.2}",
        f"{change_current:,.2f}",
        f"{chomage_current:,.2f}",
        f"{participation_current:,.2f}",
        f"{sortie_current:,.2f}",
        f"{entre_current:,.2f}"
    ],
    "Évolution par rapport à " f"{year_previous}": [
        f"{pib_percentage:.2f}%",
        f"{pib_hab_percentage:.2f}%",
        f"{export_percentage:.2f}%",
        f"{import_percentage:.2f}%",
        f"{infllation_percentage:.2f}%",
        f"{change_percentage:.2f}%",
        f"{chomage_percentage:.2f}%",
        f"{participation_percentage:.2f}%",
        f"{sortie_percentage:.2f}%",
        f"{entre_percentage:.2f}%"
    ],
        "Analyse de l'évolution": [
        analyser_evolution(pib_percentage),
        analyser_evolution(pib_hab_percentage),
        analyser_evolution(export_percentage),
        f"{analyser_evolution(import_percentage)} — {balance_commerciale}",
        analyser_evolution(infllation_percentage),
        analyser_evolution(change_percentage),
        analyser_evolution(chomage_percentage),
        analyser_evolution(participation_percentage),
        analyser_evolution(sortie_percentage),
        analyser_evolution(entre_percentage)
    ]
})

# Mise en page avec les valeurs en pourcentage
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        growth_percentage = ((difference_pib) / pib_previous) * 100
        text_color = "green" if growth_percentage >= 0 else "red"
        sign = "↑" if growth_percentage >= 0 else "↓"
        st.markdown(
    f"""
    <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
        <span style="color: black; font-size: 20px;">PIB actuel</span><br>
        <span style="color: {text_color}; font-size: 24px; font-weight: bold;">{sign} {growth_percentage:.2f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)
        
        st.write("")
        st.markdown(
    f"""
    <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
        <span style="color: black; font-size: 20px;">PIB par habitant</span><br>
        <span style="color: black; font-size: 24px; font-weight: bold;">{pib_hab_percentage:.2f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)
        st.write("")
        st.markdown(
    f"""
    <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
        <span style="color: black; font-size: 20px;">Taux de change</span><br>
        <span style="color: black; font-size: 24px; font-weight: bold;">{change_percentage:.2f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)
        st.write("")
        st.markdown(
    f"""
    <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
        <span style="color: black; font-size: 20px;">Inflation</span><br>
        <span style="color: black; font-size: 24px; font-weight: bold;">{infllation_percentage:.2f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)

    with col2:
        #interface cellule
        st.dataframe(df_indicateurs)

# Prédiction et analyse
X = filtered_data[['Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes']]
y = filtered_data['PIB']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Modèle Random Forest
rf_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=21)
rf_reg.fit(X_train, y_train)

# Prédictions
y_pred = rf_reg.predict(X_test)
filtered_data['Prévision PIB'] = rf_reg.predict(X)

# Sélection de la dernière année disponible
latest_year = filtered_data['Year'].max()
latest_data = filtered_data[filtered_data['Year'] == latest_year]

# Prévions futures
future_years = np.array(range(max_year + 1, max_year + 10)).reshape(-1, 1)

def forecast_trend(variable):
    """Prévoit la tendance d'une variable économique en utilisant une régression linéaire."""
    years = filtered_data['Year'].values.reshape(-1, 1)
    values = filtered_data[variable].values
    if np.isnan(values).any():
        raise ValueError(f"Des valeurs manquantes existent dans {variable}")
    model_trend = LinearRegression()
    model_trend.fit(years, values)
    future_values = model_trend.predict(future_years)
    noise = np.random.uniform(-0.5, 0.5, size=future_values.shape)
    return future_values + noise

variables = ['Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes']
future_exog = pd.DataFrame({var: forecast_trend(var) for var in variables})
future_forecast = rf_reg.predict(future_exog)

# forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Prévision PIB': future_forecast})
forecast_df = pd.DataFrame({
    'Year': list(filtered_data['Year']) + list(future_years.flatten()),
    'PIB': list(filtered_data['PIB']) + [np.nan] * len(future_years), 
    'Prévision PIB': list(filtered_data['Prévision PIB']) + list(future_forecast) 
})
st.write("")
st.write("")
col1, col2 = st.columns(2)
with col1:
    # Importance des variables
    st.subheader("Importance des variables")
    st.bar_chart(pd.DataFrame({'Feature': X.columns, 'Importance': rf_reg.feature_importances_}).set_index('Feature'))
with col2:
# Graphique des prévisions
    st.subheader("Evolution du PIB de Madagascar")
    plt.figure(figsize=(10, 5))
    # plt.plot(filtered_data['Year'], filtered_data['PIB'])
    plt.plot(forecast_df['Year'], forecast_df['Prévision PIB'], linestyle='-', color='blue')
    plt.title('Prévision du PIB')
    plt.xlabel('Année')
    plt.ylabel('Croissance du PIB')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
if not latest_data.empty:
    # Récupération des valeurs
    investment_entries = latest_data['Investissements directs étrangers entrées nettes'].values[0]
    investment_exits = latest_data['Investissements directs étrangers sortie nettes'].values[0]
    unemployment = latest_data['Chômage'].values[0]
    labor_participation = latest_data['Participation au marché du travail'].values[0]
    exportation = latest_data['Exportations'].values[0]
    importation = latest_data['Importations'].values[0]

st.write("")
st.write("")

col1, col2, col3 = st.columns(3)
with col1:
    # st.subheader("Investissements Directs")
    # fig1, ax1 = plt.subplots(figsize=(3, 3)) 
    # labels = ["Entrées Nettes", "Sorties Nettes"]
    # values = [investment_entries, investment_exits]
    # colors = ['#00FFFF', '#FCA311']
    # ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    # ax1.axis('equal') 
    # st.pyplot(fig1) 
    st.subheader("Investissements Directs")
    fig2, ax2 = plt.subplots(figsize=(3, 2))  
    ax2.bar(["Entrées Nettes", "Sorties Nettes"], [investment_entries, investment_exits], color=['#FCA311', '#00FFFF'], width=0.5)  
    ax2.set_ylim(0, 100)  
    ax2.set_ylabel("Taux (%)")
    st.pyplot(fig2)   
with col2:
    st.subheader("Marché du Travail")
    fig2, ax2 = plt.subplots(figsize=(3, 2))  
    ax2.bar(["Chômage", "Participation"], [unemployment, labor_participation], color=['#FCA311', '#00FFFF'], width=0.5)  
    ax2.set_ylim(0, 100)  
    ax2.set_ylabel("Taux (%)")
    st.pyplot(fig2)
with col3:
    st.subheader("Commerce")
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    labels = ["Exportations", "Importations"]
    ax2.bar(labels, [exportation, importation], color=['#FCA311', '#00FFFF'], width=0.5)  
    ax2.set_ylim(0, 100)  
    ax2.set_ylabel('Valeur')
    st.pyplot(fig2)

df_depenses['Decade'] = (df_depenses['Year'] // 10) * 10
df_recettes['Decade'] = (df_recettes['Year'] // 10) * 10
df_grouped_depenses = df_depenses.groupby('Decade')[list(indicators_depenses.values())].mean().reset_index()
df_grouped_recettes = df_recettes.groupby('Decade')[list(indicators_recettes.values())].mean().reset_index()
df_grouped_depenses.fillna(0, inplace=True)
df_grouped_recettes.fillna(0, inplace=True)

st.write("")
st.write("")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Répartition des types de dépenses")
    selected_expenses = st.multiselect("Sélectionnez les types de dépenses", list(indicators_depenses.values()), default=list(indicators_depenses.values()))
    selected_decade_depenses = st.selectbox("Sélectionnez une décennie pour les dépenses", sorted(df_grouped_depenses['Decade'].unique(), reverse=True))
    filtered_df_depenses = df_grouped_depenses[df_grouped_depenses['Decade'] == selected_decade_depenses][selected_expenses]
    if not filtered_df_depenses.empty:
        expense_values = filtered_df_depenses.iloc[0].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(selected_expenses, expense_values, color=['blue', 'green', 'red', 'orange'])
        ax.set_ylabel("Type de dépense")
        ax.set_xlabel("% du PIB")
        ax.set_title(f"Répartition des types de dépenses dans les années {selected_decade_depenses}")
        ax.grid(axis='x')
        st.pyplot(fig)
with col2:
    st.subheader("Répartition des types de recettes")
    selected_recettes = st.multiselect("Sélectionnez les types de recettes", list(indicators_recettes.values()), default=list(indicators_recettes.values()))
    selected_decade_recettes = st.selectbox("Sélectionnez une décennie pour les recettes", sorted(df_grouped_recettes['Decade'].unique(), reverse=True))
    filtered_df_recettes = df_grouped_recettes[df_grouped_recettes['Decade'] == selected_decade_recettes][selected_recettes]
    if not filtered_df_recettes.empty:
        recette_values = filtered_df_recettes.iloc[0].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(selected_recettes, recette_values, color=['blue', 'green', 'red', 'orange'])
        ax.set_ylabel("Type de recette")
        ax.set_xlabel("% du PIB")
        ax.set_title(f"Répartition des types de recettes dans les années {selected_decade_recettes}")
        ax.grid(axis='x')
        st.pyplot(fig)
#Ventilation
st.write("")
st.write("")
st.subheader("Ventilation des Dépenses")
selected_year = st.slider("Sélectionnez une année :", min_value=ventilation["Year"].min(), max_value=df["Year"].max(), value=ventilation["Year"].max())
col1, col2 = st.columns([2.7, 1.3])
with col1:
    df_melted = ventilation.melt(id_vars=["Year"], var_name="Catégorie", value_name="Valeur")
    fig_bar = px.bar(df_melted, x="Year", y="Valeur", 
                 color="Catégorie", 
                 labels={"Valeur": "% des dépenses"},
                 barmode="stack")
    st.plotly_chart(fig_bar, use_container_width=True)
with col2:
    df_sunburst = ventilation[ventilation["Year"] == selected_year].melt(id_vars=["Year"], var_name="Secteur", value_name="Montant")
    df_sunburst["Catégorie"] = "Dépenses Publiques"
    fig_sunburst = px.sunburst(df_sunburst, path=["Catégorie", "Secteur"], 
                           values="Montant",
                           title="Répartition des Dépenses Publiques",
                           color="Secteur")
    st.plotly_chart(fig_sunburst, use_container_width=True)