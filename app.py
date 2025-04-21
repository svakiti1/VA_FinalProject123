import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir = 'data'):
    admissions = pd.read_parquet(os.path.join(data_dir, 'ADMISSIONS_clean.parquet'))
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'])
    admissions['DEATHTIME'] = pd.to_datetime(admissions['DEATHTIME'])

    patients = pd.read_parquet(os.path.join(data_dir, 'PATIENTS_clean.parquet'))
    patients['DOB'] = pd.to_datetime(patients['DOB'])
    patients['DOD'] = pd.to_datetime(patients['DOD'])

    icustays = pd.read_parquet(os.path.join(data_dir, 'ICUSTAYS_clean.parquet'))
    icustays['INTIME'] = pd.to_datetime(icustays['INTIME'])
    icustays['OUTTIME'] = pd.to_datetime(icustays['OUTTIME'])

    diagnoses = pd.read_parquet(os.path.join(data_dir, 'DIAGNOSES_ICD_clean.parquet'))

    d_icd_diagnoses = pd.read_parquet(os.path.join(data_dir, 'D_ICD_DIAGNOSES_clean.parquet'))

    procedures = pd.read_parquet(os.path.join(data_dir, 'PROCEDURES_ICD_clean.parquet'))

    d_icd_procedures = pd.read_parquet(os.path.join(data_dir, 'D_ICD_PROCEDURES_clean.parquet'))

    labevents = pd.read_parquet(os.path.join(data_dir, 'LABEVENTS_clean.parquet'))
    labevents['CHARTTIME'] = pd.to_datetime(labevents['CHARTTIME'])

    d_labitems = pd.read_parquet(os.path.join(data_dir, 'D_LABITEMS_clean.parquet'))

    prescriptions = pd.read_parquet(os.path.join(data_dir, 'PRESCRIPTIONS_clean.parquet'))
    prescriptions['STARTDATE'] = pd.to_datetime(prescriptions['STARTDATE'])
    prescriptions['ENDDATE'] = pd.to_datetime(prescriptions['ENDDATE'])

    return {
        'admissions': admissions,
        'patients': patients,
        'icustays': icustays,
        'diagnoses': diagnoses,
        'd_icd_diagnoses': d_icd_diagnoses,
        'procedures': procedures,
        'd_icd_procedures': d_icd_procedures,
        'labevents': labevents,
        'd_labitems': d_labitems,
        'prescriptions': prescriptions
    }

@st.cache_data
def calculate_age(patients_df):
    now = pd.Timestamp("now")
    # Filter out invalid or null DOBs
    patients_df = patients_df[patients_df['DOB'].notnull()]
    patients_df = patients_df[patients_df['DOB'] < now]
    patients_df['AGE'] = (now - patients_df['DOB']).dt.days / 365.25
    return patients_df

@st.cache_data
def preprocess_data(data_dict):
    patients = data_dict['patients'].copy()
    admissions = data_dict['admissions'].copy()
    icustays = data_dict['icustays'].copy()
    
    patients = calculate_age(patients)
    
    merged_admissions = admissions.merge(patients[['SUBJECT_ID', 'GENDER', 'AGE']], on='SUBJECT_ID', how='left')
    merged_admissions['LOS_DAYS'] = (merged_admissions['DISCHTIME'] - merged_admissions['ADMITTIME']).dt.total_seconds() / (24 * 3600)
    
    return {**data_dict, 'patients': patients, 'merged_admissions': merged_admissions}

def create_dashboard():
    st.set_page_config(layout="wide", page_title="MIMIC-III Dashboard", page_icon="🏥")
    
    st.title("🏥 MIMIC-III Healthcare Analytics Dashboard")
    st.markdown("---")
    
    with st.spinner("Loading MIMIC-III data..."):
        data_dict = load_data()
        data_dict = preprocess_data(data_dict)
    
    nav_options = [
        "Overview & Demographics",
        "Admissions Analysis",
        "ICU Stays",
        "Diagnoses Explorer",
        "Procedures Analysis",
        "Laboratory Results",
        "Medication Analysis"
    ]
    
    with st.sidebar:
        st.image("mimic3.png")
        st.title("Navigation")
        selected_nav = st.radio("", nav_options)
        
        st.markdown("---")
        st.markdown("### Dataset Summary")
        st.markdown(f"**Patients:** {data_dict['patients'].SUBJECT_ID.nunique():,}")
        st.markdown(f"**Admissions:** {data_dict['admissions'].HADM_ID.nunique():,}")
        st.markdown(f"**ICU Stays:** {data_dict['icustays'].ICUSTAY_ID.nunique():,}")
        st.markdown("---")
        st.markdown("#### MIMIC-III Dashboard v1.0")
    
    if selected_nav == "Overview & Demographics":
        demographics_page(data_dict)
    elif selected_nav == "Admissions Analysis":
        admissions_page(data_dict)
    elif selected_nav == "ICU Stays":
        icustays_page(data_dict)
    elif selected_nav == "Diagnoses Explorer":
        diagnoses_page(data_dict)
    elif selected_nav == "Procedures Analysis":
        procedures_page(data_dict)
    elif selected_nav == "Laboratory Results":
        laboratory_page(data_dict)
    elif selected_nav == "Medication Analysis":
        medication_page(data_dict)

def demographics_page(data_dict):
    st.header("📊 Patient Demographics Overview")
    
    patients = data_dict['patients']
    admissions = data_dict['admissions']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = patients['GENDER'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        gender_counts['Gender'] = gender_counts['Gender'].map({'F': 'Female', 'M': 'Male'})
        
        fig = px.pie(gender_counts, values='Count', names='Gender', 
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution")
        age_bins = [0, 18, 30, 50, 70, 100, 200]
        age_labels = ['0-18', '19-30', '31-50', '51-70', '71-100', '100+']
        patients['age_group'] = pd.cut(patients['AGE'], bins=age_bins, labels=age_labels, right=False)
        
        age_gender = patients.groupby(['age_group', 'GENDER']).size().reset_index()
        age_gender.columns = ['Age Group', 'Gender', 'Count']
        age_gender['Gender'] = age_gender['Gender'].map({'F': 'Female', 'M': 'Male'})
        
        fig = px.bar(age_gender, x='Age Group', y='Count', color='Gender',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    barmode='group')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ethnicity Distribution")
        ethnicity_counts = admissions['ETHNICITY'].value_counts().nlargest(10).reset_index()
        ethnicity_counts.columns = ['Ethnicity', 'Count']
        
        fig = px.bar(ethnicity_counts, x='Count', y='Ethnicity', orientation='h',
                    color='Count', color_continuous_scale='Viridis')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mortality by Age Group")
        mortality = patients.groupby('age_group')['EXPIRE_FLAG'].agg(['sum', 'count']).reset_index()
        mortality['mortality_rate'] = (mortality['sum'] / mortality['count']) * 100
        mortality.columns = ['Age Group', 'Deaths', 'Total', 'Mortality Rate (%)']
        
        fig = px.line(mortality, x='Age Group', y='Mortality Rate (%)', markers=True,
                     line_shape='spline')
        fig.update_traces(line=dict(width=3))
        fig.add_bar(x=mortality['Age Group'], y=mortality['Total'], name='Total Patients',
                   marker_color='lightblue', opacity=0.3)
        fig.update_layout(height=500, yaxis_title='Count / Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Insurance & Marital Status")
    col1, col2 = st.columns(2)
    
    with col1:
        insurance_counts = admissions['INSURANCE'].value_counts().reset_index()
        insurance_counts.columns = ['Insurance', 'Count']
        
        fig = px.pie(insurance_counts, values='Count', names='Insurance',
                    color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        marital_counts = admissions['MARITAL_STATUS'].value_counts().reset_index()
        marital_counts.columns = ['Marital Status', 'Count']
        
        fig = px.bar(marital_counts, x='Marital Status', y='Count',
                    color='Count', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def admissions_page(data_dict):
    st.header("🚑 Admissions Analysis")
    
    admissions = data_dict['admissions']
    merged_admissions = data_dict['merged_admissions']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_admissions = admissions.shape[0]
        st.metric("Total Admissions", f"{total_admissions:,}")
    
    with col2:
        unique_patients = admissions['SUBJECT_ID'].nunique()
        st.metric("Unique Patients", f"{unique_patients:,}")
    
    with col3:
        readmission_rate = (1 - (unique_patients / total_admissions)) * 100
        st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Admission Types")
        admission_types = admissions['ADMISSION_TYPE'].value_counts().reset_index()
        admission_types.columns = ['Admission Type', 'Count']
        
        fig = px.bar(admission_types, x='Admission Type', y='Count',
                    color='Count', color_continuous_scale='Viridis',
                    text='Count')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Admission Locations")
        admission_locs = admissions['ADMISSION_LOCATION'].value_counts().reset_index()
        admission_locs.columns = ['Admission Location', 'Count']
        
        fig = px.pie(admission_locs, values='Count', names='Admission Location',
                    color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Length of Stay Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        los_data = merged_admissions[merged_admissions['LOS_DAYS'] < 50]
        
        fig = px.histogram(los_data, x='LOS_DAYS', nbins=50,
                          labels={'LOS_DAYS': 'Length of Stay (Days)'},
                          color_discrete_sequence=['#3366CC'])
        fig.update_layout(height=400, xaxis_title='Length of Stay (Days)', 
                         yaxis_title='Number of Admissions')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        los_by_type = merged_admissions.groupby('ADMISSION_TYPE')['LOS_DAYS'].agg(['mean', 'median']).reset_index()
        los_by_type.columns = ['Admission Type', 'Mean LOS', 'Median LOS']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=los_by_type['Admission Type'], y=los_by_type['Mean LOS'],
                           name='Mean LOS', marker_color='#3366CC'))
        fig.add_trace(go.Bar(x=los_by_type['Admission Type'], y=los_by_type['Median LOS'],
                           name='Median LOS', marker_color='#FF9900'))
        fig.update_layout(height=400, barmode='group',
                         xaxis_title='Admission Type', yaxis_title='Length of Stay (Days)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Discharge Destinations")
    
    discharge_locs = admissions['DISCHARGE_LOCATION'].value_counts().nlargest(10).reset_index()
    discharge_locs.columns = ['Discharge Location', 'Count']
    
    fig = px.bar(discharge_locs, x='Discharge Location', y='Count',
                color='Count', color_continuous_scale='Viridis',
                text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def icustays_page(data_dict):
    st.header("🏥 ICU Stays Analysis")
    
    icustays = data_dict['icustays']
    patients = data_dict['patients']
    
    icustays['LOS_DAYS'] = icustays['LOS']
    
    icu_patients = patients[patients['SUBJECT_ID'].isin(icustays['SUBJECT_ID'])]
    merged_icu = icustays.merge(icu_patients[['SUBJECT_ID', 'GENDER', 'AGE']], on='SUBJECT_ID', how='left')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_icu_stays = icustays.shape[0]
        st.metric("Total ICU Stays", f"{total_icu_stays:,}")
    
    with col2:
        unique_icu_patients = icustays['SUBJECT_ID'].nunique()
        st.metric("Unique ICU Patients", f"{unique_icu_patients:,}")
    
    with col3:
        avg_los = icustays['LOS'].mean()
        st.metric("Avg Length of Stay", f"{avg_los:.1f} days")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ICU Unit Distribution")
        care_unit_counts = icustays['FIRST_CAREUNIT'].value_counts().reset_index()
        care_unit_counts.columns = ['ICU Unit', 'Count']
        
        fig = px.bar(care_unit_counts, x='ICU Unit', y='Count',
                    color='ICU Unit', text='Count')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ICU Length of Stay Distribution")
        los_data = icustays[icustays['LOS'] < 20]
        
        fig = px.histogram(los_data, x='LOS', nbins=40,
                          color_discrete_sequence=['#3366CC'],
                          labels={'LOS': 'Length of Stay (Days)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("ICU Length of Stay by Age Group and Gender")
    
    age_bins = [0, 18, 30, 50, 70, 100, 200]
    age_labels = ['0-18', '19-30', '31-50', '51-70', '71-100', '100+']
    merged_icu['age_group'] = pd.cut(merged_icu['AGE'], bins=age_bins, labels=age_labels, right=False)
    
    los_by_age_gender = merged_icu.groupby(['age_group', 'GENDER'])['LOS'].mean().reset_index()
    los_by_age_gender.columns = ['Age Group', 'Gender', 'Average LOS']
    los_by_age_gender['Gender'] = los_by_age_gender['Gender'].map({'F': 'Female', 'M': 'Male'})
    
    fig = px.bar(los_by_age_gender, x='Age Group', y='Average LOS', color='Gender',
                barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ICU Unit by Gender")
        unit_gender = merged_icu.groupby(['FIRST_CAREUNIT', 'GENDER']).size().reset_index()
        unit_gender.columns = ['ICU Unit', 'Gender', 'Count']
        unit_gender['Gender'] = unit_gender['Gender'].map({'F': 'Female', 'M': 'Male'})
        
        fig = px.bar(unit_gender, x='ICU Unit', y='Count', color='Gender',
                    barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average LOS by ICU Unit")
        los_by_unit = merged_icu.groupby('FIRST_CAREUNIT')['LOS'].mean().reset_index()
        los_by_unit.columns = ['ICU Unit', 'Average LOS']
        los_by_unit = los_by_unit.sort_values('Average LOS', ascending=False)
        
        fig = px.bar(los_by_unit, x='ICU Unit', y='Average LOS',
                    color='Average LOS', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def diagnoses_page(data_dict):
    st.header("🩺 Diagnoses Explorer")
    
    diagnoses = data_dict['diagnoses']
    d_icd_diagnoses = data_dict['d_icd_diagnoses']
    patients = data_dict['patients']
    
    merged_diagnoses = diagnoses.merge(d_icd_diagnoses, on='ICD9_CODE', how='left')
    
    st.subheader("Top Diagnoses")
    
    top_n = st.slider("Select number of top diagnoses to view:", 5, 20, 10)
    
    diagnosis_counts = merged_diagnoses['SHORT_TITLE'].value_counts().nlargest(top_n).reset_index()
    diagnosis_counts.columns = ['Diagnosis', 'Count']
    
    fig = px.bar(diagnosis_counts, x='Count', y='Diagnosis', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(height=600, xaxis_title='Number of Patients', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Diagnoses by Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_diagnoses = merged_diagnoses['SHORT_TITLE'].value_counts().nlargest(5).index.tolist()
        selected_diagnosis = st.selectbox("Select diagnosis to analyze:", top_diagnoses)
        
        diagnosis_patients = merged_diagnoses[merged_diagnoses['SHORT_TITLE'] == selected_diagnosis]['SUBJECT_ID'].unique()
        diagnosis_patient_data = patients[patients['SUBJECT_ID'].isin(diagnosis_patients)]
        
        age_bins = [0, 18, 30, 50, 70, 100, 200]
        age_labels = ['0-18', '19-30', '31-50', '51-70', '71-100', '100+']
        diagnosis_patient_data['age_group'] = pd.cut(diagnosis_patient_data['AGE'], bins=age_bins, labels=age_labels, right=False)
        
        age_dist = diagnosis_patient_data['age_group'].value_counts().reset_index()
        age_dist.columns = ['Age Group', 'Count']
        
        fig = px.bar(age_dist, x='Age Group', y='Count',
                    color='Count', color_continuous_scale='Viridis',
                    title=f"Age Distribution for {selected_diagnosis}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        gender_dist = diagnosis_patient_data['GENDER'].value_counts().reset_index()
        gender_dist.columns = ['Gender', 'Count']
        gender_dist['Gender'] = gender_dist['Gender'].map({'F': 'Female', 'M': 'Male'})
        
        fig = px.pie(gender_dist, values='Count', names='Gender',
                    title=f"Gender Distribution for {selected_diagnosis}",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Co-occurring Diagnoses")
    
    patients_with_selected = merged_diagnoses[merged_diagnoses['SHORT_TITLE'] == selected_diagnosis]['SUBJECT_ID'].unique()
    
    cooccurring = merged_diagnoses[merged_diagnoses['SUBJECT_ID'].isin(patients_with_selected)]
    cooccurring = cooccurring[cooccurring['SHORT_TITLE'] != selected_diagnosis]
    
    cooccur_counts = cooccurring['SHORT_TITLE'].value_counts().nlargest(10).reset_index()
    cooccur_counts.columns = ['Diagnosis', 'Count']
    
    fig = px.bar(cooccur_counts, x='Count', y='Diagnosis', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                title=f"Top Co-occurring Diagnoses with {selected_diagnosis}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def procedures_page(data_dict):
    st.header("🔪 Procedures Analysis")
    
    procedures = data_dict['procedures']
    d_icd_procedures = data_dict['d_icd_procedures']
    
    merged_procedures = procedures.merge(d_icd_procedures, on='ICD9_CODE', how='left')
    
    st.subheader("Top Procedures")
    
    top_n = st.slider("Select number of top procedures to view:", 5, 20, 10)
    
    procedure_counts = merged_procedures['SHORT_TITLE'].value_counts().nlargest(top_n).reset_index()
    procedure_counts.columns = ['Procedure', 'Count']
    
    fig = px.bar(procedure_counts, x='Count', y='Procedure', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(height=600, xaxis_title='Number of Patients', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Procedures by Diagnosis")
    
    diagnoses = data_dict['diagnoses']
    d_icd_diagnoses = data_dict['d_icd_diagnoses']
    merged_diagnoses = diagnoses.merge(d_icd_diagnoses, on='ICD9_CODE', how='left')
    
    top_diagnoses = merged_diagnoses['SHORT_TITLE'].value_counts().nlargest(10).index.tolist()
    selected_diagnosis = st.selectbox("Select diagnosis to view related procedures:", top_diagnoses)
    
    diagnosis_hadm_ids = merged_diagnoses[merged_diagnoses['SHORT_TITLE'] == selected_diagnosis]['HADM_ID'].unique()
    procedures_for_diagnosis = merged_procedures[merged_procedures['HADM_ID'].isin(diagnosis_hadm_ids)]
    
    proc_counts = procedures_for_diagnosis['SHORT_TITLE'].value_counts().nlargest(10).reset_index()
    proc_counts.columns = ['Procedure', 'Count']
    
    fig = px.bar(proc_counts, x='Count', y='Procedure', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                title=f"Top Procedures for Patients with {selected_diagnosis}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Procedure Combinations")
    
    procedure_pairs = []
    for hadm_id in merged_procedures['HADM_ID'].unique()[:10000]:
        procs = merged_procedures[merged_procedures['HADM_ID'] == hadm_id]['SHORT_TITLE'].tolist()
        # Convert all elements to strings to ensure consistent comparison
        procs = [str(proc) for proc in procs]
        if len(procs) >= 2:
            for i in range(len(procs)):
                for j in range(i+1, len(procs)):
                    # Now safely compare as strings
                    if procs[i] < procs[j]:
                        procedure_pairs.append((procs[i], procs[j]))
                    else:
                        procedure_pairs.append((procs[j], procs[i]))
    
    pair_counts = pd.Series(procedure_pairs).value_counts().nlargest(10).reset_index()
    pair_counts.columns = ['Procedure Pair', 'Count']
    pair_counts['Procedure 1'] = pair_counts['Procedure Pair'].apply(lambda x: x[0])
    pair_counts['Procedure 2'] = pair_counts['Procedure Pair'].apply(lambda x: x[1])
    pair_counts['Pair Label'] = pair_counts['Procedure 1'] + ' & ' + pair_counts['Procedure 2']
    
    fig = px.bar(pair_counts, x='Count', y='Pair Label', orientation='h',
                color='Count', color_continuous_scale='Viridis')
    fig.update_layout(height=600, yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

def laboratory_page(data_dict):
    st.header("🧪 Laboratory Results Analysis")
    
    labevents = data_dict['labevents']
    d_labitems = data_dict['d_labitems']
    
    merged_lab = labevents.merge(d_labitems, on='ITEMID', how='left')
    
    st.subheader("Laboratory Test Categories")
    
    category_counts = d_labitems['CATEGORY'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    fig = px.pie(category_counts, values='Count', names='Category',
                color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Laboratory Test Explorer")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_category = st.selectbox("Select lab category:", 
                                        sorted(d_labitems['CATEGORY'].unique()))
        
        category_items = d_labitems[d_labitems['CATEGORY'] == selected_category]
        
        test_counts = merged_lab[merged_lab['CATEGORY'] == selected_category]['LABEL'].value_counts().nlargest(30)
        
        selected_test = st.selectbox("Select lab test:", 
                                    test_counts.index.tolist())
    
    with col2:
        selected_lab_data = merged_lab[(merged_lab['LABEL'] == selected_test) & 
                                      (merged_lab['VALUENUM'].notnull()) &
                                      (merged_lab['VALUENUM'] < merged_lab['VALUENUM'].quantile(0.99)) &
                                      (merged_lab['VALUENUM'] > merged_lab['VALUENUM'].quantile(0.01))]
        
        units = selected_lab_data['VALUEUOM'].mode()[0] if not selected_lab_data.empty else ""
        
        fig = px.histogram(selected_lab_data, x='VALUENUM', nbins=50,
                          labels={'VALUENUM': f'Value ({units})'},
                          title=f"Distribution of {selected_test} values")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        normal_abnormal = selected_lab_data['FLAG'].value_counts().reset_index()
        normal_abnormal.columns = ['Flag', 'Count']
        if not normal_abnormal.empty:
            fig = px.pie(normal_abnormal, values='Count', names='Flag',
                        title=f"Distribution of Normal vs Abnormal Results for {selected_test}",
                        color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Abnormal Lab Results by ICU Unit")
    
    selected_lab_data = merged_lab[merged_lab['LABEL'] == selected_test]
    
    icustays = data_dict['icustays']
    patients = data_dict['patients']
    
    lab_with_patient = selected_lab_data.merge(patients[['SUBJECT_ID', 'GENDER']], on='SUBJECT_ID', how='left')
    lab_with_icu = lab_with_patient.merge(icustays[['SUBJECT_ID', 'HADM_ID', 'FIRST_CAREUNIT']], 
                                        on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    abnormal_by_unit = lab_with_icu[lab_with_icu['FLAG'].notnull()].groupby(['FIRST_CAREUNIT', 'FLAG']).size().reset_index()
    abnormal_by_unit.columns = ['ICU Unit', 'Flag', 'Count']
    
    if not abnormal_by_unit.empty:
        fig = px.bar(abnormal_by_unit, x='ICU Unit', y='Count', color='Flag',
                    barmode='group', title=f"Abnormal {selected_test} Results by ICU Unit")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient data to show abnormal results by ICU unit for this test.")

def medication_page(data_dict):
    st.header("💊 Medication Analysis")
    
    prescriptions = data_dict['prescriptions']
    
    st.subheader("Most Common Medications")
    
    top_n = st.slider("Select number of top medications to view:", 5, 20, 10)
    
    med_counts = prescriptions['DRUG_NAME_GENERIC'].value_counts().nlargest(top_n).reset_index()
    med_counts.columns = ['Medication', 'Count']
    
    fig = px.bar(med_counts, x='Count', y='Medication', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                text='Count')
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(height=600, xaxis_title='Number of Prescriptions', yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Medication Routes")
    
    route_counts = prescriptions['ROUTE'].value_counts().nlargest(10).reset_index()
    route_counts.columns = ['Route', 'Count']
    
    fig = px.pie(route_counts, values='Count', names='Route',
                color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Medication Analysis by Diagnosis")
    
    diagnoses = data_dict['diagnoses']
    d_icd_diagnoses = data_dict['d_icd_diagnoses']
    merged_diagnoses = diagnoses.merge(d_icd_diagnoses, on='ICD9_CODE', how='left')
    
    top_diagnoses = merged_diagnoses['SHORT_TITLE'].value_counts().nlargest(10).index.tolist()
    selected_diagnosis = st.selectbox("Select diagnosis to view related medications:", top_diagnoses, key="med_diag")
    
    diagnosis_hadm_ids = merged_diagnoses[merged_diagnoses['SHORT_TITLE'] == selected_diagnosis]['HADM_ID'].unique()
    meds_for_diagnosis = prescriptions[prescriptions['HADM_ID'].isin(diagnosis_hadm_ids)]
    
    med_counts = meds_for_diagnosis['DRUG_NAME_GENERIC'].value_counts().nlargest(10).reset_index()
    med_counts.columns = ['Medication', 'Count']
    
    fig = px.bar(med_counts, x='Count', y='Medication', orientation='h',
                color='Count', color_continuous_scale='Viridis',
                title=f"Top Medications for Patients with {selected_diagnosis}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Medication Usage by ICU Unit")
    
    icustays = data_dict['icustays']
    
    top_meds = prescriptions['DRUG_NAME_GENERIC'].value_counts().nlargest(5).index.tolist()
    selected_med = st.selectbox("Select medication to analyze by ICU unit:", top_meds)
    
    med_icu = prescriptions[prescriptions['DRUG_NAME_GENERIC'] == selected_med].merge(
        icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'FIRST_CAREUNIT']],
        on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'], how='inner'
    )
    
    med_by_unit = med_icu.groupby('FIRST_CAREUNIT').size().reset_index()
    med_by_unit.columns = ['ICU Unit', 'Count']
    
    fig = px.bar(med_by_unit, x='ICU Unit', y='Count',
                color='Count', color_continuous_scale='Viridis',
                title=f"Usage of {selected_med} by ICU Unit")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Medication Dosage Analysis")
    
    med_data = prescriptions[prescriptions['DRUG_NAME_GENERIC'] == selected_med].copy()
    
    common_units = med_data['DOSE_UNIT_RX'].value_counts().nlargest(1).index[0] if not med_data.empty else None
    
    if common_units:
        unit_data = med_data[med_data['DOSE_UNIT_RX'] == common_units].copy()
        unit_data['DOSE_VAL_RX'] = pd.to_numeric(unit_data['DOSE_VAL_RX'], errors='coerce')
        
        unit_data = unit_data[
            (unit_data['DOSE_VAL_RX'].notnull()) & 
            (unit_data['DOSE_VAL_RX'] < unit_data['DOSE_VAL_RX'].quantile(0.99)) &
            (unit_data['DOSE_VAL_RX'] > unit_data['DOSE_VAL_RX'].quantile(0.01))
        ]
        
        fig = px.histogram(unit_data, x='DOSE_VAL_RX', nbins=50,
                         title=f"Distribution of {selected_med} Dosages ({common_units})")
        fig.update_layout(height=400, xaxis_title=f'Dosage ({common_units})')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Insufficient dosage data available for {selected_med}")

if __name__ == "__main__":
    create_dashboard()