import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- הגדרות ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "project_analysis_final.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Final_Visuals_V5_Renamed") 

os.makedirs(os.path.join(OUTPUT_DIR, "Tables"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Graphs"), exist_ok=True)

# רשימות
LEXICAL_PAIRS = [
    ('ילד', 'תינוק'), ('שופט', 'דיין'), ('שפה', 'לשון'), ('סיר', 'קדרה'),
    ('צמד', 'זוג'), ('אהבה', 'חיבה'), ('בטן', 'כרס'), ('גם', 'אף'),
    ('גבול', 'תחום'), ('עם', 'אומה'), ('ריב', 'קטטה'), ('סיבה', 'עילה')
]
SUB_WORDS = ['ש', 'אשר', 'כי', 'כאשר', 'מאשר', 'אם', 'פן']

# רשימות לניקוי
PUNCTUATION = ['.', ',', ':', ';', '-', '"', "'", '(', ')', '?', '!', '׃', 'O', 'PUNCT', '-','[BLANK]','־', 'nan', 'UNK']

# הסרנו את 0, 06, 07 מרשימת המחיקה כי אנחנו מתקנים אותם
JUNK_CATEGORIES = ['nan', 'UNK', 'None', 'NULL', '']

def rev_heb(text):
    """היפוך טקסט עברי בלבד"""
    text = str(text)
    if any("\u0590" <= c <= "\u05EA" for c in text):
        return text[::-1]
    return text

def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, "Tables", f"{name}.csv")
    df.to_csv(path, encoding='utf-8-sig', index=False)
    print(f"✅ Saved Table: {name}.csv")

# --- פונקציות גרפיקה ---

def plot_bar_with_values(df, x_col, y_col, title_heb, title_eng_suffix, filename):
    plt.figure(figsize=(14, 8))
    
    # סינון
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[x_col, y_col])
    df_clean[x_col] = df_clean[x_col].astype(str).str.strip()
    df_clean = df_clean[~df_clean[x_col].isin(JUNK_CATEGORIES)]
    
    if df_clean.empty: return

    # היפוך עברית בתוויות ציר ה-X
    df_clean['Label_Rev'] = df_clean[x_col].apply(rev_heb)
    
    ax = sns.barplot(data=df_clean, x='Label_Rev', y=y_col, palette='viridis')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10)

    full_title = f"{rev_heb(title_heb)} {title_eng_suffix}"
    
    plt.title(full_title, fontsize=16, pad=20)
    plt.ylabel("")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", filename), dpi=150)
    plt.close()
    print(f"📊 Saved Graph: {filename}")

def plot_top3_separate_per_category(df):
    print("\n--- Generating Top 3 Words Graphs ---")
    
    df_filtered = df[~df['Major_Category'].isin(JUNK_CATEGORIES)]
    
    counts = df_filtered.groupby(['Major_Category', 'Lemma']).size().reset_index(name='Count')
    counts = counts.sort_values(['Major_Category', 'Count'], ascending=[True, False])
    
    target_categories = ['mikra', 'hazal', 'modern']
    
    for category in target_categories:
        df_cat = counts[counts['Major_Category'] == category]
        if df_cat.empty: continue
        
        top3 = df_cat.head(3).copy()
        top3['Word_Rev'] = top3['Lemma'].apply(rev_heb)
        
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(data=top3, x='Word_Rev', y='Count', palette='viridis')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=11, fontweight='bold')

        plt.title(f"{rev_heb('שלושת המילים הנפוצות')} - {category}", fontsize=18)
        plt.xlabel("")
        plt.ylabel(rev_heb("מספר מופעים"), fontsize=12)
        plt.xticks(fontsize=14)
        plt.tight_layout()
        
        filename = f"Top3_Words_{category}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", filename), dpi=150)
        plt.close()

def plot_lexical_comparison_per_category(df):
    print("\n--- Generating Lexical Comparison Graphs ---")
    target_categories = ['mikra', 'hazal', 'modern']
    
    for category in target_categories:
        df_cat = df[df['Major_Category'] == category]
        if len(df_cat) == 0: continue
        
        word_counts = df_cat['Lemma'].value_counts()
        
        plot_data = []
        for w1, w2 in LEXICAL_PAIRS:
            c1 = word_counts.get(w1, 0)
            c2 = word_counts.get(w2, 0)
            plot_data.append({'Word': w1, 'Count': c1, 'PairID': f"{w1}/{w2}"})
            plot_data.append({'Word': w2, 'Count': c2, 'PairID': f"{w1}/{w2}"})
            
        df_plot = pd.DataFrame(plot_data)
        if df_plot['Count'].sum() == 0: continue

        df_plot['Label_Rev'] = df_plot['Word'].apply(rev_heb)

        plt.figure(figsize=(16, 9))
        colors = ['#1f77b4', '#aec7e8'] * len(LEXICAL_PAIRS)
        
        ax = sns.barplot(data=df_plot, x='Label_Rev', y='Count', palette=colors)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10)
            
        plt.title(f"{rev_heb('השוואת זוגות מילים')} - {category}", fontsize=20)
        plt.xlabel("")
        plt.ylabel(rev_heb("מספר מופעים"), fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        
        for i in range(1, len(LEXICAL_PAIRS)):
            plt.axvline(x=i*2 - 0.5, color='gray', linestyle=':', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", f"Lexical_Compare_{category}.png"), dpi=150)
        plt.close()

def plot_subordination_frequency_per_category(df):
    print("\n--- Generating Subordination Frequency Graphs ---")
    target_categories = ['mikra', 'hazal', 'modern']
    
    for category in target_categories:
        df_cat = df[df['Major_Category'] == category]
        if len(df_cat) == 0: continue
        
        word_counts = df_cat['Lemma'].value_counts()
        plot_data = []
        for word in SUB_WORDS:
            count = word_counts.get(word, 0)
            plot_data.append({'Word': rev_heb(word), 'Count': count})
        df_plot = pd.DataFrame(plot_data)
        if df_plot['Count'].sum() == 0: continue

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=df_plot, x='Word', y='Count', palette='viridis')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', padding=3, fontsize=10)
            
        full_title = f"{rev_heb('שכיחות מילות שעבוד')} - {category}"
        plt.title(full_title, fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", f"Subordination_Freq_{category}.png"), dpi=150)
        plt.close()

def plot_subordination_percent_bars(df):
    print("\n--- Generating Subordination Percentages ---")
    target_categories = ['mikra', 'hazal', 'modern']
    
    for category in target_categories:
        df_cat = df[df['Major_Category'] == category]
        if len(df_cat) == 0: continue
        
        word_counts = df_cat['Lemma'].value_counts()
        total_sub_words = sum([word_counts.get(w, 0) for w in SUB_WORDS])
        
        if total_sub_words == 0: continue
        
        plot_data = []
        for word in SUB_WORDS:
            count = word_counts.get(word, 0)
            if count > 0:
                percent = (count / total_sub_words) * 100
                plot_data.append({
                    'Word': rev_heb(word), 
                    'Percentage': percent
                })
        
        df_plot = pd.DataFrame(plot_data)
        df_plot = df_plot.sort_values('Percentage', ascending=False)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_plot, y='Word', x='Percentage', palette='pastel', orient='h')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3)
            
        full_title = f"{rev_heb('התפלגות מילות שעבוד')} (%) - {category}"
        plt.title(full_title, fontsize=18)
        plt.xlabel("%")
        plt.ylabel("")
        plt.tight_layout()
        
        plt.savefig(os.path.join(OUTPUT_DIR, "Graphs", f"Subordination_Percents_{category}.png"), dpi=150)
        plt.close()

def process_level(df, group_col, suffix, title_suffix_eng):
    print(f"\n--- Processing: {suffix} ---")
    
    df = df[df[group_col].notna()]
    df = df[~df[group_col].astype(str).str.strip().isin(JUNK_CATEGORIES)]

    # טבלאות
    counts = df.groupby([group_col, 'Lemma']).size().reset_index(name='Count')
    counts = counts.sort_values([group_col, 'Count'], ascending=[True, False])
    for cat in df[group_col].unique():
        cat_clean = str(cat).replace('/', '_')
        top = counts[counts[group_col] == cat].head(1000)
        save_csv(top, f"Freq_Words_{suffix}_{cat_clean}")

    # טבלאות השוואה
    pair_rows = []
    for cat in df[group_col].unique():
        sub_df = df[df[group_col] == cat]
        size = len(sub_df)
        if size == 0: continue
        row = {group_col: cat}
        for w1, w2 in LEXICAL_PAIRS:
            c1 = len(sub_df[sub_df['Lemma'] == w1])
            c2 = len(sub_df[sub_df['Lemma'] == w2])
            row[w1] = (c1 / size) * 10000
            row[w2] = (c2 / size) * 10000
        pair_rows.append(row)
    if pair_rows:
        cols = [group_col] + [w for pair in LEXICAL_PAIRS for w in pair]
        pairs_df = pd.DataFrame(pair_rows)
        existing_cols = [c for c in cols if c in pairs_df.columns]
        save_csv(pairs_df[existing_cols], f"Lexical_Pairs_{suffix}")

    # מילות שעבוד
    sub_rows = []
    for cat in df[group_col].unique():
        sub_df = df[df[group_col] == cat]
        size = len(sub_df)
        if size == 0: continue
        row = {group_col: cat}
        for w in SUB_WORDS:
            c = len(sub_df[sub_df['Lemma'] == w])
            row[w] = (c / size) * 10000
        sub_rows.append(row)
    if sub_rows:
        save_csv(pd.DataFrame(sub_rows), f"Subordination_{suffix}")

    # גרפים ראשיים
    print("   Calculating stats...")
    sent_stats = df.groupby([group_col, 'File_Name', 'Sentence_ID']).agg({
        'Lemma': 'count',       
        'Sentence_Depth': 'max' 
    }).rename(columns={'Lemma': 'Real_Length'}).reset_index()
    
    unique_stats = df.groupby([group_col, 'File_Name', 'Sentence_ID'])['Lemma'].nunique().reset_index(name='Unique_Words')
    final_stats = pd.merge(sent_stats, unique_stats, on=[group_col, 'File_Name', 'Sentence_ID'])
    
    for col in ['Real_Length', 'Sentence_Depth', 'Unique_Words']:
        final_stats[col] = pd.to_numeric(final_stats[col], errors='coerce')
    
    final_stats = final_stats.dropna(subset=['Real_Length', 'Sentence_Depth', 'Unique_Words'])

    cols_to_avg = ['Real_Length', 'Sentence_Depth', 'Unique_Words']
    avgs = final_stats.groupby(group_col)[cols_to_avg].mean().reset_index()
    
    plot_bar_with_values(avgs, group_col, 'Real_Length', "ממוצע אורך משפט", title_suffix_eng, f"Graph_Length_{suffix}.png")
    plot_bar_with_values(avgs, group_col, 'Sentence_Depth', "ממוצע עומק תחבירי", title_suffix_eng, f"Graph_Depth_{suffix}.png")
    plot_bar_with_values(avgs, group_col, 'Unique_Words', "ממוצע מילים ייחודיות", title_suffix_eng, f"Graph_Unique_{suffix}.png")

def main():
    print(f"📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False, dtype=str)
    except:
        print("❌ Error reading file.")
        return

    print("🧹 Cleaning Data...")
    df = df[~df['Lemma'].isin(PUNCTUATION)]
    df['Lemma'] = df['Lemma'].str.strip()
    df = df[df['Lemma'].str.len() > 0]
    
    df['Sentence_Depth'] = pd.to_numeric(df['Sentence_Depth'], errors='coerce')
    
    df['Source_Category'] = df['Source_Category'].astype(str).str.lower().str.strip()
    if 'Major_Category' not in df.columns: df['Major_Category'] = np.nan
    
    df.loc[df['Source_Category'].str.contains('hazal'), 'Major_Category'] = 'hazal'
    df.loc[df['Source_Category'].str.contains('modern'), 'Major_Category'] = 'modern'
    df.loc[df['Source_Category'].str.contains('mikra'), 'Major_Category'] = 'mikra'
    
    if 'Sub_Category' not in df.columns: df['Sub_Category'] = 'General'
    df['Sub_Category'] = df['Sub_Category'].fillna('General')
    
    df = df.dropna(subset=['Major_Category'])
    df = df[~df['Major_Category'].isin(['nan', ''])]

    # --- התיקון שביקשת: שינוי שמות הקטגוריות הבעייתיות ---
    print("🔧 Renaming & Merging categories ('0' -> 'haaretz', '06/07' -> 'medical')...")
    
    # 1. החלפת "0" ב-"haaretz" רק במודרני
    mask_modern = df['Major_Category'] == 'modern'
    
    # משתמשים ב-str() כדי לוודא שתופסים גם מספר וגם מחרוזת
    df.loc[mask_modern & (df['Sub_Category'].astype(str).str.strip() == '0'), 'Sub_Category'] = 'haaretz'
    
    # 2. איחוד 06 ו-07 ל-"medical"
    medical_mask = df['Sub_Category'].astype(str).str.strip().isin(['06', '6', '07', '7'])
    df.loc[mask_modern & medical_mask, 'Sub_Category'] = 'medical'

    print("📊 Valid Major Categories:", df['Major_Category'].unique())

    # 1. דוחות ראשיים
    process_level(df, 'Major_Category', 'Major', "(Major)")

    # 2. דוחות תתי קטגוריות (Modern only)
    print("\n--- Filtering only Modern Hebrew for Sub-Category Graphs ---")
    df_modern = df[df['Major_Category'] == 'modern'].copy()
    if not df_modern.empty:
        print("   Sub-Categories in Modern (Cleaned):", df_modern['Sub_Category'].unique())
        process_level(df_modern, 'Sub_Category', 'Sub_Modern', "(Sub - Modern)")
    else:
        print("⚠️ No Modern data found!")

    # 3. גרפים מיוחדים לפי תקופות
    plot_lexical_comparison_per_category(df)
    plot_subordination_frequency_per_category(df)
    plot_subordination_percent_bars(df)
    plot_top3_separate_per_category(df)

    print(f"\n✅ DONE! All charts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()