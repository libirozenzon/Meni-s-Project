import pandas as pd
import os
import numpy as np

# --- הגדרות ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "project_analysis_final.csv")

# 1. רשימת הזוגות הלקסיקליים (עדית דורון, עמוד 4)
LEXICAL_PAIRS = [
    ('ילד', 'תינוק'),    # yeled / tinoq
    ('שופט', 'דיין'),    # šopet / dayan
    ('שפה', 'לשון'),     # sapa / lašon
    ('סיר', 'קדרה'),     # sir / qdera
    ('צמד', 'זוג'),      # cemed / zug
    ('אהבה', 'חיבה'),    # Pahaba / hiba
    ('בטן', 'כרס'),      # beten / keres
    ('גם', 'אף'),        # gam / Pap
    ('גבול', 'תחום'),    # gbul / thum
    ('עם', 'אומה'),      # Lam / Puma
    ('ריב', 'קטטה'),     # rib / qlata
    ('סיבה', 'עילה')     # siba / Gila
]

# 2. מילות שעבוד וקישור (עדית דורון, דוגמאות 5-20)
SUBORDINATION_WORDS = [
    'אשר', 'ש',       # Relative
    'כי', 'יען',      # Causal
    'אם', 'לו', 'אילו', # Conditional
    'פן',             # Avertive
    'למען',           # Purpose
    'כאשר', 'מאשר',   # Comparative/Temporal
    'בלתי'            # Exceptive
]

def analyze_statistics():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: הקובץ המאוחד לא נמצא. הרץ קודם את הסקריפטים של החילוץ והאיחוד.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # ניקוי ושימוש בעמודה הקיימת
    df = df.dropna(subset=['Lemma', 'POS'])
    df['Lemma'] = df['Lemma'].astype(str)
    
    if 'Sub_Category' not in df.columns:
        print("⚠️ Warning: Sub_Category column missing. Running logic on General categories only.")
        df['Sub_Category'] = df['Source_Category']
    
    # מילוי ערכים ריקים ב-'Other'
    df['Sub_Category'] = df['Sub_Category'].fillna('Other')

    print(f"\n--- Total Data: {len(df)} words ---")
    
    # חישוב גודל כל תת-מאגר (לצורך נרמול)
    sub_corpus_sizes = df['Sub_Category'].value_counts()
    
    # ==========================================
    # 1. סטטיסטיקות משפטים (אורך ועומק)
    # ==========================================
    print("\n📊 1. Calculating Sentence Stats (Length & Depth)...")
    sentences = df.groupby(['Sub_Category', 'Sentence_ID']).agg({
        'Sentence_Length': 'first',
        'Sentence_Depth': 'first'
    }).reset_index()

    sent_stats = sentences.groupby('Sub_Category')[['Sentence_Length', 'Sentence_Depth']].describe()
    sent_stats.to_csv(os.path.join(SCRIPT_DIR, "stats_1_sentences_sub_corpus.csv"))
    print("   Saved: stats_1_sentences_sub_corpus.csv")

    # ==========================================
    # 2. התפלגות קטגוריות מילים (POS Categories)
    # ==========================================
    # דרישה: "קטגוריות המילים (שם עצם, פועל...)"
    print("\n📊 2. Calculating POS Categories Distribution...")
    pos_counts = df.groupby(['Sub_Category', 'POS']).size().unstack(fill_value=0)
    # נרמול לאחוזים (כדי להשוות בין טקסטים בגדלים שונים)
    pos_ratios = pos_counts.div(pos_counts.sum(axis=1), axis=0)
    pos_ratios.to_csv(os.path.join(SCRIPT_DIR, "stats_2_pos_distribution.csv"))
    print("   Saved: stats_2_pos_distribution.csv")

    # ==========================================
    # 3. שכיחות מילים (Top Words)
    # ==========================================
    print("\n📊 3. Calculating Top Words...")
    words_only = df[~df['POS'].isin(['PUNCT', 'X', 'NUM'])]
    top_words = words_only.groupby(['Sub_Category', 'Lemma']).size().reset_index(name='Count')
    top_words = top_words.sort_values(['Sub_Category', 'Count'], ascending=[True, False])
    top_words.groupby('Sub_Category').head(20).to_csv(os.path.join(SCRIPT_DIR, "stats_3_top_words.csv"))
    print("   Saved: stats_3_top_words.csv")

    # ==========================================
    # 4. זוגות לקסיקליים (דורון, עמ' 4)
    # ==========================================
    print("\n📊 4. Analyzing Lexical Pairs (Doron Page 4)...")
    target_words = [word for pair in LEXICAL_PAIRS for word in pair]
    filtered = df[df['Lemma'].isin(target_words)]
    
    word_counts = filtered.groupby(['Sub_Category', 'Lemma']).size().unstack(fill_value=0)
    # נרמול ל-10,000 מילים
    norm_counts = word_counts.div(sub_corpus_sizes, axis=0) * 10000
    norm_counts.to_csv(os.path.join(SCRIPT_DIR, "stats_4_doron_lexical_pairs.csv"))
    print("   Saved: stats_4_doron_lexical_pairs.csv")

    # ==========================================
    # 5. מילות שעבוד (דוגמאות 5-20)
    # ==========================================
    print("\n📊 5. Analyzing Subordination Words...")
    sub_filtered = df[df['Lemma'].isin(SUBORDINATION_WORDS)]
    sub_counts = sub_filtered.groupby(['Sub_Category', 'Lemma']).size().unstack(fill_value=0)
    sub_norm = sub_counts.div(sub_corpus_sizes, axis=0) * 10000
    sub_norm.to_csv(os.path.join(SCRIPT_DIR, "stats_5_subordination.csv"))
    print("   Saved: stats_5_subordination.csv")

    # ==========================================
    # 6. מבני V1 (משפט פותח בפועל)
    # ==========================================
    print("\n📊 6. Analyzing V1 Structures (Verb-First)...")
    first_words = df.groupby(['Sub_Category', 'Sentence_ID']).first().reset_index()
    first_words['Is_Verb_First'] = first_words['POS'] == 'VERB'
    v1_stats = first_words.groupby('Sub_Category')['Is_Verb_First'].mean()
    v1_stats.to_csv(os.path.join(SCRIPT_DIR, "stats_6_v1_structures.csv"))
    print("   Saved: stats_6_v1_structures.csv")

    # ==========================================
    # 7. שימוש ב"של" (לעומת סמיכות)
    # ==========================================
    print("\n📊 7. Analyzing 'Shel' usage...")
    shel_counts = df[df['Lemma'] == 'של'].groupby('Sub_Category').size()
    shel_norm = (shel_counts / sub_corpus_sizes) * 10000
    shel_norm.to_csv(os.path.join(SCRIPT_DIR, "stats_7_possessives_shel.csv"))
    print("   Saved: stats_7_possessives_shel.csv")

    print("\n✅ Analysis Complete! 7 report files are ready.")

if __name__ == "__main__":
    analyze_statistics()