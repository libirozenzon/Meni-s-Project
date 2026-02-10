import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- הגדרות ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "project_analysis_final.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Final_Linguistic_Analysis") 

os.makedirs(os.path.join(OUTPUT_DIR, "Tables"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Graphs"), exist_ok=True)

# רשימות לניקוי
PUNCTUATION = ['.', ',', ':', ';', '-', '"', "'", '(', ')', '?', '!', '׃', 'O', 'PUNCT', '-','[BLANK]','־', 'nan', 'UNK']
JUNK_CATEGORIES = ['nan', 'UNK', 'None', 'NULL', '']

# רשימת "זבל" לסינון מגרף חלקי הדיבור
POS_FILTER = ['O', 'PUNCT', 'yyDOT', 'yyCM', 'yyQUOT', 'UNK', 'nan', '']

def rev_heb(text):
    text = str(text)
    if any("\u0590" <= c <= "\u05EA" for c in text):
        return text[::-1]
    return text

# --- פונקציות גרפיקה משופרות ---

def plot_pos_distribution(df):
    """
    גרף התפלגות חלקי דיבור - ללא O וללא סימני פיסוק
    """
    print("   📊 Generating POS Distribution Graph (Cleaned)...")
    
    # סינון אגרסיבי של כל מה שלא חלק דיבור אמיתי
    df_clean = df[~df['POS'].isin(POS_FILTER)].copy()
    
    if df_clean.empty:
        print("      ⚠️ No valid POS tags found after filtering.")
        return

    # חישוב אחוזים
    pos_counts = df_clean.groupby(['Major_Category', 'POS']).size().reset_index(name='Count')
    
    # חישוב סך הכל לכל קטגוריה (מתוך המילים המזוהות בלבד!)
    totals = pos_counts.groupby('Major_Category')['Count'].transform('sum')
    pos_counts['Percentage'] = (pos_counts['Count'] / totals) * 100
    
    # סינון זנבות (פחות מ-1%)
    pos_counts = pos_counts[pos_counts['Percentage'] > 1]
    
    plt.figure(figsize=(12, 7))
    
    pivot_df = pos_counts.pivot(index='Major_Category', columns='POS', values='Percentage')
    pivot_df = pivot_df.fillna(0)
    
    pivot_df.plot(kind='bar', stacked=True, colormap='tab20', figsize=(12, 7))
    
    plt.title(rev_heb("התפלגות חלקי דיבור (מתוך מילים מזוהות בלבד)"), fontsize=18)
    plt.xlabel("")
    plt.ylabel("%")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", "Linguistic_POS_Distribution.png"), dpi=150)
    plt.close()

def plot_verb_tense_distribution(df):
    """
    גרף זמנים - ניסיון לתפוס את המקרא גם אם התיוג לא מושלם
    """
    print("   📊 Generating Verb Tense Distribution Graph...")
    
    # ניסיון ראשון: קח כל מה שמסומן כפועל
    # ניסיון שני: קח כל מה שיש לו 'Tense' (גם אם ה-POS הוא O בטעות)
    
    # בדיקה: האם יש בכלל Tense למקרא?
    mikra_check = df[(df['Major_Category'] == 'mikra') & (df['Tense'].notna())]
    if mikra_check.empty:
        print("      ⚠️ WARNING: No Tense data found for Mikra in the CSV. The graph will likely be empty for Mikra.")
    
    # סינון: או שה-POS הוא פועל, או שיש מידע על זמן (Tense) והוא לא ריק
    verbs = df[
        (df['POS'] == 'VERB') | 
        ((df['Tense'].notna()) & (~df['Tense'].isin(['', 'nan', 'UNK', 'O'])))
    ].copy()
    
    if verbs.empty: return

    # ניקוי זמנים לא חוקיים
    verbs = verbs[~verbs['Tense'].isin(['', 'nan', 'UNK', 'O'])]
    
    # חישוב אחוזים
    tense_counts = verbs.groupby(['Major_Category', 'Tense']).size().reset_index(name='Count')
    totals = tense_counts.groupby('Major_Category')['Count'].transform('sum')
    tense_counts['Percentage'] = (tense_counts['Count'] / totals) * 100
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=tense_counts, x='Major_Category', y='Percentage', hue='Tense', palette='Set2')
    
    for container in ax.containers if 'ax' in locals() else []:
        pass # רק למנוע שגיאה אם אין ax

    plt.title(rev_heb("התפלגות זמנים (מתוך מילים בעלות נטיית זמן)"), fontsize=18)
    plt.xlabel("")
    plt.ylabel("%")
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", "Linguistic_Verb_Tenses.png"), dpi=150)
    plt.close()

def plot_v1_structure(df):
    print("   📊 Generating V1 Structure Analysis...")
    # לוקחים את המילה הראשונה בכל משפט
    first_words = df.groupby(['Major_Category', 'File_Name', 'Sentence_ID']).head(1).copy()
    
    # בדיקה מורחבת: האם זה פועל? (לפי POS או לפי קיום זמן)
    first_words['Is_Verb_First'] = (first_words['POS'] == 'VERB') | (first_words['Tense'].notna() & (first_words['Tense'] != ''))
    
    v1_stats = first_words.groupby('Major_Category')['Is_Verb_First'].mean().reset_index()
    v1_stats['Percentage'] = v1_stats['Is_Verb_First'] * 100
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=v1_stats, x='Major_Category', y='Percentage', palette='coolwarm')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
        
    plt.title(rev_heb("אחוז המשפטים המתחילים בפועל (V1)"), fontsize=18)
    plt.xlabel("")
    plt.ylabel("%")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", "Linguistic_Structure_V1.png"), dpi=150)
    plt.close()

def plot_shel_usage(df):
    print("   📊 Generating 'Shel' Analysis...")
    results = []
    for cat in df['Major_Category'].unique():
        df_cat = df[df['Major_Category'] == cat]
        
        # הרחבת הגדרת שם עצם (כולל O אם יש לו יידוע אולי? לא, נשאר עם NOUN)
        # אם אין תיוג NOUN למקרא, הגרף הזה יהיה בעייתי למקרא
        num_nouns = len(df_cat[df_cat['POS'] == 'NOUN'])
        num_shel = len(df_cat[df_cat['Lemma'] == 'של'])
        
        ratio = (num_shel / num_nouns * 100) if num_nouns > 0 else 0
        results.append({'Major_Category': cat, 'Shel_Ratio': ratio})
        
    stats_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=stats_df, x='Major_Category', y='Shel_Ratio', palette='magma')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.title(rev_heb("יחס השימוש ב'של' (לכל 100 שמות עצם)"), fontsize=18)
    plt.xlabel("")
    plt.ylabel(rev_heb("יחס"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", "Linguistic_Possessive_Shel.png"), dpi=150)
    plt.close()

def main():
    print(f"📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False, dtype=str)
    except:
        print("❌ Error reading file.")
        return

    print("🧹 Preparing Data for Linguistic Analysis...")
    # ניקוי בסיסי
    df = df[~df['Lemma'].isin(PUNCTUATION)]
    df['Lemma'] = df['Lemma'].str.strip()
    
    # סידור קטגוריות
    df['Source_Category'] = df['Source_Category'].astype(str).str.lower().str.strip()
    if 'Major_Category' not in df.columns: df['Major_Category'] = np.nan
    
    df.loc[df['Source_Category'].str.contains('hazal'), 'Major_Category'] = 'hazal'
    df.loc[df['Source_Category'].str.contains('modern'), 'Major_Category'] = 'modern'
    df.loc[df['Source_Category'].str.contains('mikra'), 'Major_Category'] = 'mikra'
    
    # ניקוי קטגוריות ריקות
    df = df.dropna(subset=['Major_Category'])
    df = df[~df['Major_Category'].isin(JUNK_CATEGORIES)]

    # --- הפעלת התיקון לקטגוריות מודרניות (כדי שיהיה תואם לגרפים האחרים) ---
    print("🔧 Applying category fixes (haaretz/medical)...")
    mask_modern = df['Major_Category'] == 'modern'
    if 'Sub_Category' in df.columns:
        df.loc[mask_modern & (df['Sub_Category'].astype(str).str.strip() == '0'), 'Sub_Category'] = 'haaretz'
        medical_mask = df['Sub_Category'].astype(str).str.strip().isin(['06', '6', '07', '7'])
        df.loc[mask_modern & medical_mask, 'Sub_Category'] = 'medical'

    print("📊 Valid Categories:", df['Major_Category'].unique())

    # הרצת הניתוחים
    plot_pos_distribution(df)
    plot_verb_tense_distribution(df)
    plot_v1_structure(df)
    plot_shel_usage(df)

    print(f"\n✅ DONE! Updated linguistic charts in: {OUTPUT_DIR}/Graphs")

if __name__ == "__main__":
    main()