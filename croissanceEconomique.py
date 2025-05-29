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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Prévision du PIB", layout="wide", initial_sidebar_state="expanded")
# Indicateurs économiques
def get_live_wbdata():
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
    df.rename(columns={'date': 'Année'}, inplace=True)
    df = df.dropna(subset=['PIB','Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes'], how='all')
    df['Année'] = df['Année'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)
    # completer les valeurs Nan
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.iloc[::-1]
    df_depenses = wbdata.get_dataframe(indicators_depenses, country="MDG")
    df_depenses.reset_index(inplace=True)
    df_depenses.rename(columns={'date': 'Année'}, inplace=True)
    df_depenses = df_depenses.dropna(subset=[ "Dépenses publiques","Rémunération des employés","Dépenses nationales brutes","Dépenses de consommation finale"], how='all')
    df_depenses['Année'] = df_depenses['Année'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)
    df_depenses.fillna(0, inplace=True)

    df_recettes = wbdata.get_dataframe(indicators_recettes, country="MDG")
    df_recettes.reset_index(inplace=True)
    df_recettes.rename(columns={'date': 'Année'}, inplace=True)
    df_recettes = df_recettes.dropna(subset=["Recettes fiscales","Recettes, hors subventions","Impôts sur le revenu"], how='all')
    df_recettes['Année'] = df_recettes['Année'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)
    df_recettes.fillna(0, inplace=True)

    ventilation = wbdata.get_dataframe(indicators_ventilation_depenses, country="MDG") 
    ventilation.reset_index(inplace=True)
    ventilation.rename(columns={'date': 'Année'}, inplace=True)
    ventilation = ventilation.dropna(subset=["Paiements d'intérêts","Dépenses militaires","Dépenses de santé","Dépenses publiques d'éducation","Dépenses courantes d'éducation",], how='all')
    ventilation['Année'] = ventilation['Année'].apply(lambda x: int(x.replace("YR", "")) if isinstance(x, str) else x)
    ventilation.fillna(0, inplace=True)
    
# Titre
    min_Année, max_Année = df['Année'].min(), df['Année'].max()
    selected_Années = st.sidebar.slider("Sélectionner une plage d'années", min_Année, max_Année, (min_Année, max_Année), 1)
    st.title("Analyse de la croissance économique")
    filtered_data = df[(df['Année'] >= selected_Années[0]) & (df['Année'] <= selected_Années[1])]
    filtered_data.dropna(inplace=True)

# Mise en page avec les valeurs en pourcentage
    st.write("")

# Fonction pour l'analyse
    Année_current = selected_Années[1]
    Année_previous = Année_current - 1
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
      
    pib_previous = filtered_data['PIB'].dropna().iloc[-2]
    pib_current = filtered_data['PIB'].dropna().iloc[-1]
    difference_pib = pib_current - pib_previous
    pib_percentage = ((difference_pib) / pib_previous)*100

    pib_hab_current = filtered_data['PIB par habitant'].dropna().iloc[-1]
    pib_hab_previous = filtered_data['PIB par habitant'].dropna().iloc[-2]
    difference_pib_hab = pib_hab_current - pib_hab_previous
    pib_hab_percentage = ((difference_pib_hab) / pib_hab_previous)*100
    
    infllation_current = filtered_data['Inflation'].dropna().iloc[-1]
    infllation_previous = filtered_data['Inflation'].dropna().iloc[-2]
    difference_infllation = infllation_current - infllation_previous
    infllation_percentage = ((difference_infllation) / infllation_previous)*100

    change_current = filtered_data['Taux de change officiel'].dropna().iloc[-1]
    change_previous = filtered_data['Taux de change officiel'].dropna().iloc[-2]
    difference_change = change_current - change_previous
    change_percentage = ((difference_change) / change_previous)*100

    export_current = filtered_data['Exportations'].dropna().iloc[-1]
    export_previous = filtered_data['Exportations'].dropna().iloc[-2]
    difference_export = export_current - export_previous
    export_percentage = ((difference_export) / export_previous)*100

    import_current = filtered_data['Importations'].dropna().iloc[-1]
    import_previous = filtered_data['Importations'].dropna().iloc[-2]
    difference_import = import_current - import_previous
    import_percentage = ((difference_import) / import_previous)*100

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

    df_indicateurs = pd.DataFrame({
        "Indicateur": ["PIB", "PIB par habitant", "Exportations", "Importations","Inflation","Taux de change officiel","Chômage","Participation au marché du travail","Investissements directs étrangers sortie nettes","Investissements directs étrangers entrées nettes"],
        f"Valeur en {Année_current}": [
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
        "Évolution par rapport à " f"{Année_previous}": [
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

# Prédiction et analyse
    X = filtered_data[['Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes']]
    y = filtered_data['PIB']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Modèle Random Forest
    rf_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=21)
    rf_reg.fit(X_train, y_train)

# Prédictions
    filtered_data['Prévision PIB'] = rf_reg.predict(X)

# Sélection de la dernière année disponible
    latest_Année = filtered_data['Année'].max()
    latest_data = filtered_data[filtered_data['Année'] == latest_Année]

# Prévions futures
    future_Années = np.array(range(max_Année + 1, max_Année + 10)).reshape(-1, 1)

    def forecast_trend(variable):
        """Prévoit la tendance d'une variable économique en utilisant une régression linéaire."""
        Années = filtered_data['Année'].values.reshape(-1, 1)
        values = filtered_data[variable].values
        if np.isnan(values).any():
            raise ValueError(f"Des valeurs manquantes existent dans {variable}")
        model_trend = LinearRegression()
        model_trend.fit(Années, values)
        future_values = model_trend.predict(future_Années)
        noise = np.random.uniform(-0.5, 0.5, size=future_values.shape)
        return future_values + noise

    variables = ['Exportations', 'PIB par habitant', 'Participation au marché du travail', 'Taux de change officiel', 'Dépenses publiques', 'Chômage', 'Investissements directs étrangers entrées nettes', 'Investissements directs étrangers sortie nettes']
    future_exog = pd.DataFrame({var: forecast_trend(var) for var in variables})
    future_forecast = rf_reg.predict(future_exog)

# forecast_df = pd.DataFrame({'Année': future_Années.flatten(), 'Prévision PIB': future_forecast})
    forecast_df = pd.DataFrame({
        'Année': list(filtered_data['Année']) + list(future_Années.flatten()),
        'PIB': list(filtered_data['PIB']) + [np.nan] * len(future_Années), 
        'Prévision PIB': list(filtered_data['Prévision PIB']) + list(future_forecast) 
    })
# Mise en page avec les valeurs en pourcentage
    col1,col2,col3,col4=st.columns(4)
    with col1:
            if pib_percentage > 0:
                text_color = "#15EE4B"  
                sign = "↑"
            elif pib_percentage < 0:
                text_color = "red"
                sign = "↓"
            else:
                text_color = "orange"
                sign = ""
            st.markdown(
        f"""
        <div style="background-color: #063970; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
            <span style="color: #ffffff; font-size: 20px;">PIB actuel</span><br>
            <span style="color: {text_color}; font-size: 24px; font-weight: bold;">{sign} {pib_percentage:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    with col2:
            if pib_hab_percentage > 0:
                text_color = "#15EE4B"
                sign = "↑"
            elif pib_hab_percentage < 0:
                text_color = "red"
                sign = "↓"
            else:
                text_color = "orange"
                sign = ""
            st.markdown(
        f"""
        <div style="background-color: #063970; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
            <span style="color: #ffffff; font-size: 20px;">PIB par habitant</span><br>
            <span style="color: {text_color}; font-size: 24px; font-weight: bold;">{sign} {pib_hab_percentage:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    with col3:
            if change_percentage > 0:
                text_color = "#15EE4B" 
                sign = "↑"
            elif change_percentage < 0:
                text_color = "red"
                sign = "↓"
            else:
                text_color = "orange"
                sign = "" #→
            st.markdown(
        f"""
        <div style="background-color: #063970; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
            <span style="color: #ffffff; font-size: 20px;">Taux de change</span><br>
            <span style="color: {text_color}; font-size: 24px; font-weight: bold;">{sign} {change_percentage:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    with col4:
            if infllation_percentage > 0:
                text_color = "#15EE4B"
                sign = "↑"
            elif infllation_percentage < 0:
                text_color = "red"
                sign = "↓"
            else:
                text_color = "orange"
                sign = ""
            st.markdown(
        f"""
        <div style="background-color: #063970; padding: 10px; border-radius: 5px; text-align: center;box-shadow: 0px 4px 6px rgba(0,0,0,0.3);">
            <span style="color: #ffffff; font-size: 20px;">Inflation</span><br>
            <span style="color: {text_color}; font-size: 24px; font-weight: bold;">{sign}  {infllation_percentage:.2f}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.write("")
    st.write("")
    with st.container():
        if not latest_data.empty:
    # Récupération des valeurs
            unemployment = latest_data['Chômage'].values[0]
            labor_participation = latest_data['Participation au marché du travail'].values[0]
            exportation = latest_data['Exportations'].values[0]
            importation = latest_data['Importations'].values[0]
        col1, col2 = st.columns([1, 3])
        with col1:
            fig2, ax2 = plt.subplots(figsize=(3, 2),facecolor="#2368B3") 
            labels = ["Chômage", "Participation"]
            ax2.bar(labels, [unemployment, labor_participation], color=['#063970'], width=0.5)  
            ax2.set_ylim(0, 100)  
            ax2.set_ylabel("Taux (%)",color="#ffffff")
            ax2.set_title("Marché du Travail",color="#ffffff")
            ax2.set_facecolor("#2368B3")
            ax2.tick_params(axis='y', colors='#ffffff')
            ax2.tick_params(axis='x', colors='#ffffff')
            st.pyplot(fig2)
            st.write("")
            st.write("")
            fig2, ax2 = plt.subplots(figsize=(3, 2),facecolor="#2368B3")
            labels = ["Exportations", "Importations"]
            ax2.bar(labels, [exportation, importation], color=['#063970'], width=0.5)  
            ax2.set_ylim(0, 100)  
            ax2.set_ylabel('Valeur',color="#ffffff")
            ax2.set_title("Commerce",color="#ffffff")
            ax2.set_facecolor("#2368B3")
            ax2.tick_params(axis='y', colors='#ffffff')
            ax2.tick_params(axis='x', colors='#ffffff')
            st.pyplot(fig2)

        with col2:
            # plt.plot(filtered_data['Année'], filtered_data['PIB'])
            fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="#2368B3")
            ax.plot(forecast_df['Année'], forecast_df['Prévision PIB'], linestyle='-', color='#063970')
            ax.set_title('Évolution du PIB de Madagascar et sa prévision de 10 ans', color="#ffffff")
            ax.set_ylabel('Valeur', color="#ffffff")
            ax.tick_params(axis='y', colors='#ffffff')
            ax.tick_params(axis='x', colors='#ffffff')
            ax.set_facecolor('#2368B3')
            ax.grid(axis='y')
            st.pyplot(fig)
            st.write("")
            st.write("")
    col1,col2=st.columns([2,1])
    with col1:
        st.subheader("Analyse de quelques indicateurs économiques")
        gb = GridOptionsBuilder.from_dataframe(df_indicateurs)
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        grid_options = gb.build()
        grid_response = AgGrid(
            df_indicateurs,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            theme='blue'
        )
        selected_rows = grid_response['selected_rows']
        df_pivot = df_indicateurs.set_index("Indicateur").T
    with col2:
        if selected_rows is not None and not selected_rows.empty:
            selected_indic = selected_rows.iloc[0]['Indicateur']
            st.write(f"Graphique d'évolution pour {selected_indic} (10 dernières années)")
            df = filtered_data.sort_values(by='Année')
            last_10_years = sorted(df['Année'].unique())[-10:]
            df_last_10 = filtered_data[filtered_data['Année'].isin(last_10_years)]
            fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#2368B3")
            ax.set_facecolor("#2368B3")
            ax.plot(df_last_10['Année'], df_last_10[selected_indic], color="#063970")
            ax.set_ylabel("Valeur", color="#ffffff")
            ax.tick_params(axis='y', colors='#ffffff')
            ax.tick_params(axis='x', colors='#ffffff')
            ax.grid(axis='y')
            st.pyplot(fig)
        else:
            st.write("")
            st.write("Clique sur une ligne du tableau pour voir le graphique")
        st.write("")
        st.write("")
        
    df_depenses['Decade'] = (df_depenses['Année'] // 10) * 10
    df_recettes['Decade'] = (df_recettes['Année'] // 10) * 10
    df_grouped_depenses = df_depenses.groupby('Decade')[list(indicators_depenses.values())].mean().reset_index()
    df_grouped_recettes = df_recettes.groupby('Decade')[list(indicators_recettes.values())].mean().reset_index()
    df_grouped_depenses.fillna(0, inplace=True)
    df_grouped_recettes.fillna(0, inplace=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Répartition de types de dépenses")
        selected_expenses = st.sidebar.multiselect("Sélectionnez les types de dépenses", list(indicators_depenses.values()), default=list(indicators_depenses.values()))
        selected_decade_depenses = st.selectbox("Sélectionnez une décennie pour les dépenses", sorted(df_grouped_depenses['Decade'].unique(), reverse=True))
        filtered_df_depenses = df_grouped_depenses[df_grouped_depenses['Decade'] == selected_decade_depenses][selected_expenses]
        if not filtered_df_depenses.empty:
            expense_values = filtered_df_depenses.iloc[0].values
            fig, ax = plt.subplots(figsize=(10, 6),facecolor="#2368B3")
            ax.barh(selected_expenses, expense_values, color=['#063970'])
            ax.set_ylabel("Type de dépense",color="#ffffff")
            ax.set_xlabel("Dépenses en moyenne",color="#ffffff")
            ax.set_title(f"Répartition de types de dépenses dans les années {selected_decade_depenses}",color="#ffffff")
            ax.grid(axis='x')
            ax.set_facecolor("#2368B3")
            ax.tick_params(axis='x', colors='#ffffff')
            ax.tick_params(axis='y', colors='#ffffff')
            st.pyplot(fig)
    with col2:
        st.subheader("Répartition de types de recettes")
        selected_recettes = st.sidebar.multiselect("Sélectionnez les types de recettes", list(indicators_recettes.values()), default=list(indicators_recettes.values()))
        selected_decade_recettes = st.selectbox("Sélectionnez une décennie pour les recettes", sorted(df_grouped_recettes['Decade'].unique(), reverse=True))
        filtered_df_recettes = df_grouped_recettes[df_grouped_recettes['Decade'] == selected_decade_recettes][selected_recettes]
        if not filtered_df_recettes.empty:
            recette_values = filtered_df_recettes.iloc[0].values
            fig, ax = plt.subplots(figsize=(10, 6),facecolor="#2368B3")
            ax.barh(selected_recettes, recette_values, color=['#063970'])
            ax.set_ylabel("Type de recette",color="#ffffff")
            ax.set_xlabel(f"Recettes en moyenne",color="#ffffff")
            ax.set_title(f"Répartition de types de recettes dans les années {selected_decade_recettes}",color="#ffffff")
            ax.grid(axis='x')
            ax.set_facecolor("#2368B3")
            ax.tick_params(axis='x', colors='#ffffff')
            ax.tick_params(axis='y', colors='#ffffff')
            st.pyplot(fig)
#Ventilation
    st.write("")
    st.write("")
    st.subheader("Evolution des Dépenses Publiques")
    col1, col2 = st.columns([2.7, 1.3])
    with col1:
        min_Année_ventillation, max_Année_ventillation = ventilation['Année'].min(), ventilation['Année'].max()
        selected_Année_ventillation = st.slider("Sélectionner une plage d'années", min_Année_ventillation, max_Année_ventillation, (min_Année_ventillation, max_Année_ventillation), 1)
        filtered_ventillation = ventilation[(ventilation['Année'] >= selected_Année_ventillation[0]) & (ventilation['Année'] <= selected_Année_ventillation[1])]
        filtered_ventillation.dropna(inplace=True)
        st.plotly_chart(px.line(filtered_ventillation.melt(id_vars=["Année"], var_name="Catégorie", value_name="Valeur"), x="Année", y="Valeur", color="Catégorie", labels={"Valeur": "Dépenses"}), use_container_width=True)
    with col2:
        selected_Année = st.number_input(
            f"Entrez une année (valeur entre {int(ventilation['Année'].min())} et {int(ventilation['Année'].max())}) :", 
            min_value=int(ventilation["Année"].min()), 
            max_value=int(ventilation["Année"].max()), 
            value=int(ventilation["Année"].max()), 
            step=1
            )
            # Préparer les données pour le sunburst
        df_sunburst = ventilation[ventilation["Année"] == selected_Année].melt(
            id_vars=["Année"], var_name="Secteur", value_name="Montant"
        )
        df_wide = df_sunburst.pivot(index="Année", columns="Secteur", values="Montant").reset_index()
# Ajouter une colonne de catégorie (racine du sunburst)
        df_sunburst["Catégorie"] = "Dépenses Publiques"

# Trier les secteurs par montant décroissant
        df_sunburst = df_sunburst.sort_values(by="Montant", ascending=False)

# Créer le graphique sunburst
        fig_sunburst = px.sunburst(
            df_sunburst,
            path=["Catégorie", "Secteur"],
            values="Montant",
            title="Répartition des Dépenses Publiques par secteur",
            color="Secteur",
            color_discrete_sequence=px.colors.qualitative.Pastel 
        )
        
        fig_sunburst.update_traces(
            hovertemplate='<b>%{label}</b><br>Montant : %{value:.2f} unités<extra></extra>'
        )

# Afficher le graphique dans Streamlit
        st.plotly_chart(fig_sunburst, use_container_width=True)

# Analyse textuelle automatique
        top_depense = df_sunburst
        if not df_sunburst.empty:
    # S'assurer qu'il y a bien des données valides pour Montant
            if df_sunburst['Montant'].notnull().any():
                top_depense = df_sunburst.loc[df_sunburst['Montant'].idxmax()]
                st.markdown(f"En **{selected_Année}**, la plus grande dépense publique a concerné le secteur **{top_depense['Secteur']}**, avec un montant de **{top_depense['Montant']}**.")
            else:
             st.warning(f"Aucune valeur disponible pour l'année {selected_Année}.")
        else:
            st.warning(f"Aucune donnée pour l'année {selected_Année}.")

    return df_pivot,forecast_df, filtered_df_depenses, filtered_df_recettes, filtered_ventillation,df_wide

data = get_live_wbdata()
