import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('Gb_mod_Salry.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

data = load_model()
Gb_mod = data['model']

def show_predict_pg():
    st.markdown(
    "<h1 style='color:grey; font-size: 24px; font-weight: bold; font-style: italic;'>💰 Techpro <span style='color:#BD7E58; font-size: 32px; font-weight: bold; font-style: italic;'>Salary App</h1>", 
    unsafe_allow_html=True
)

    st.write("""###### Kindly enter details""")

    variable_mappings = {
        'OrgSize': {'2 to 9': 0, '10 to 19': 1, '20 to 99': 2, '100 to 499': 3, '500 to 999': 4, '1,000 to 4,999': 5,
                    '5,000 to 9,999': 6, '10,000 & above': 7},
        'WorkExp': {'0 to 5': 0, '6 to 10': 1, '11 to 15': 2, '16 to 20': 3, '21 to 25': 4, '26 to 30': 5, '31 & above': 6}
    }

    le_DevType = data['le_DevType']
    le_Country = data['le_Country']
    le_Employment = data['le_Employment']
    le_Worktype = data['le_Worktype']

    Dev_Type = (
        "full-stack",
        "back-end",
        "front-end",
        "desktop/enterprise app",
        "mobile",
        "Engineering manager",
        "embedded applications",
        "Data scientist",
        "DevOps specialist",
        "Data engineer",
        "Cloud engineer",
        "others"
    )

    Countries = (
        'United States of America',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Canada',
        'India',
        'France',
        'Netherlands',
        'Poland',
        'Brazil',
        'Australia',
        'Spain',
        'Sweden',
        'Italy',
        'Switzerland',
        'Austria',
        'Denmark',
        'Czech Republic',
        'Norway',
        'Portugal',
        'Israel',
        'Belgium',
        'Finland',
        'Russian Federation',
        'Ukraine',
        'New Zealand',
        'Romania',
        'others'
    )

    Employment_type = (
        'full-time',
        'Contract',
        'part-time',
        'others'
    )

    Org_Size = (
        '2 to 9',
        '10 to 19',
        '20 to 99',
        '100 to 499',
        '500 to 999',
        '1,000 to 4,999',
        '5,000 to 9,999',
        '10,000 & above',
        'others'
    )

    Work_Exp = (
        '0 to 5',
        '6 to 10',
        '11 to 15',
        '16 to 20',
        '21 to 25',
        '26 to 30',
        '31 & above'
    )

    Work_type = (
        'Remote',
        'Hybrid',
        'Onsite'
    )

    DevType = st.selectbox('Developer Type', Dev_Type)
    Country = st.selectbox('Country', Countries)
    Employment = st.selectbox('Employment type', Employment_type)
    OrgSize = st.selectbox('Organization size', Org_Size)
    WorkExp = st.selectbox('Years of experience', Work_Exp)
    Worktype = st.selectbox('Work Type', Work_type)
    YearsCodePro = st.slider("Years of professional coding experience", 0, 50, 2)

    ok = st.button("Predict Salary")
    if ok:
        user_input = np.array(
            [[DevType, Country, Employment, OrgSize, WorkExp, Worktype, YearsCodePro]])

        user_input[:, 3] = np.vectorize(variable_mappings['OrgSize'].get, otypes=[int])(user_input[:, 3])
        user_input[:, 4] = np.vectorize(variable_mappings['WorkExp'].get, otypes=[int])(user_input[:, 4])
        user_input[:, 0] = le_DevType.transform(user_input[:, 0])
        user_input[:, 1] = le_Country.transform(user_input[:, 1])
        user_input[:, 2] = le_Employment.transform(user_input[:, 2])
        user_input[:, 5] = le_Worktype.transform(user_input[:, 5])

        Salary = Gb_mod.predict(user_input)

        formatted_salary = "{:,.0f}".format(Salary[0])

        # Output results
        st.subheader(f'The estimated Salary ${formatted_salary}')

# Run the app
if __name__ == '__main__':
    show_predict_pg()
