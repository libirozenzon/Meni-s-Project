import pandas as pd
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "project_analysis_final.csv")

def prepare_weka_final():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: הקובץ המאוחד לא נמצא.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # 1. ניקוי והכנת נתונים
    print("Cleaning data...")
    df['Lemma'] = df['Lemma'].fillna('').astype(str)
    df['POS'] = df['POS'].fillna('UNK').astype(str)
    df['Gender'] = df['Gender'].fillna('').astype(str)
    df['Tense'] = df['Tense'].fillna('').astype(str)
    df['Binyan'] = df['Binyan'].fillna('').astype(str)
    # וידוא שקיים Sub_Category, אם לא ממלאים ברירת מחדל
    if 'Sub_Category' not in df.columns:
        df['Sub_Category'] = 'General'
    df['Sub_Category'] = df['Sub_Category'].fillna('General').astype(str)

    # יצירת "מילה מורפולוגית" (למשל: VERB_Past_Masc)
    df['Morph_Tag'] = df['POS'] + "_" + df['Gender'] + "_" + df['Tense'] + "_" + df['Binyan']
    df['Morph_Tag'] = df['Morph_Tag'].str.replace('__', '_').str.strip('_')

    # 2. קיבוץ למשפטים (Grouping)
    print("Grouping words into sentences...")
    # מוסיפים את Sub_Category ל-groupby כדי שיישמר ברמת המשפט
    sentences = df.groupby(['Source_Category', 'Sub_Category', 'File_Name', 'Sentence_ID']).agg({
        'Lemma': lambda x: " ".join(x),       # מחבר את המילים למשפט
        'Morph_Tag': lambda x: " ".join(x),   # מחבר את התגיות למשפט
        'Sentence_Depth': 'first',            # לוקח את עומק המשפט
        'Sentence_Length': 'first'            # לוקח את אורך המשפט
    }).reset_index()

    # סינון משפטים קצרים מדי (פחות מ-2 מילים) - רעש
    sentences = sentences[sentences['Lemma'].str.len() > 2]

    # שינוי שם העמודה ל-class (ככה Weka אוהבת)
    sentences.rename(columns={'Source_Category': 'class_label'}, inplace=True)

    print(f"Total sentences: {len(sentences)}")

    # 3. פיצול ל-Train ו-Test
    # Train = מקרא + חז"ל
    train_df = sentences[sentences['class_label'].isin(['mikra', 'hazal'])].copy()
    
    # Test = מודרני
    test_df = sentences[sentences['class_label'] == 'modern'].copy()
    
    # 4. שמירה
    print("Saving files for Weka...")
    
    # שומרים גם את Sub_Category כדי שנוכל לפלטר ב-Excel או Weka אחר כך
    cols = ['Lemma', 'Morph_Tag', 'Sentence_Depth', 'Sentence_Length', 'Sub_Category', 'class_label']
    
    train_df[cols].to_csv(os.path.join(SCRIPT_DIR, "weka_train.csv"), index=False)
    test_df[cols].to_csv(os.path.join(SCRIPT_DIR, "weka_test.csv"), index=False)

    print("\n✅ DONE!")
    print(f"Train Set (Mikra/Hazal): {len(train_df)} sentences -> weka_train.csv")
    print(f"Test Set (Modern):       {len(test_df)} sentences  -> weka_test.csv")
    print("הערה: העמודה Sub_Category נשמרה בקבצים כדי לאפשר ניתוח נפרד לחדשות/ספרות וכו'.")

if __name__ == "__main__":
    prepare_weka_final()