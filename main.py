import os
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# --- הגדרות ---
INPUT_FILE = "input.txt"      # שם קובץ הטקסט שלך
OUTPUT_FILE = "results.xlsx"  # שם קובץ האקסל שייווצר

def create_sample_file_if_missing():
    """
    פונקציית עזר: אם אין קובץ טקסט, יוצרת אחד לדוגמה כדי שהקוד ירוץ.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"Creating sample file: {INPUT_FILE}...")
        sample_text = """
        בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת.
        ראש הממשלה בנימין נתניהו נפגש עם נשיא ארה"ב ג'ו ביידן בבית הלבן.
        חברת מובילאיי ממוקמת בירושלים והיא מובילה בתחום הרכב האוטונומי.
        """
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(sample_text.strip())

def load_dictabert():
    """
    שלב 1: טעינת המודל והטוקנייזר של DictaBERT
    """
    print("Loading DictaBERT model... (This might take a minute the first time)")
    tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')
    # trust_remote_code=True חובה כי המודל מכיל קוד מותאם אישית
    model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)
    
    model.eval() # מעבר למצב חיזוי (לא אימון)
    
    # בדיקה אם יש כרטיס מסך (GPU) לשיפור ביצועים, אחרת שימוש במעבד
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully on: {device}")
    
    return model, tokenizer

def load_and_split_data(filepath):
    """
    שלב 2: טעינת הקובץ וחלוקה למשפטים
    """
    print(f"Reading data from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # חלוקה גסה למשפטים לפי נקודה או ירידת שורה.
    # אפשר לשפר את זה בעתיד, אבל זה מספיק להתחלה.
    sentences = [s.strip() for s in text.replace('\n', '.').split('.') if s.strip()]
    print(f"Found {len(sentences)} sentences to process.")
    return sentences

def extract_entities(prediction):
    """
    שלב 4: לוגיקה לחילוץ המידע שמעניין אותנו מתוך הפלט של המודל
    """
    extracted_data = []
    
    # התוצאה היא רשימה (למרות ששלחנו משפט אחד), לכן לוקחים את הראשון
    result = prediction[0] 
    original_text = result['text']
    
    # בדיקה אם נמצאו ישויות (NER)
    if result['ner_entities']:
        for entity in result['ner_entities']:
            extracted_data.append({
                'Original Sentence': original_text,
                'Entity Text': entity['phrase'],
                'Entity Type': entity['label'] # למשל: PER, ORG, LOC, TIMEX
            })
    else:
        # אם לא נמצא כלום, אפשר לשמור שורה ריקה או לדלג
        pass
        
    return extracted_data

def main():
    # 0. בדיקה שיש קובץ לעבוד איתו
    create_sample_file_if_missing()
    
    # 1. טעינת המודל
    model, tokenizer = load_dictabert()
    
    # 2. הכנת הדאטה
    sentences = load_and_split_data(INPUT_FILE)
    
    all_results = []
    
    # 3. הרצה בלולאה (שימוש ב-tqdm כדי לראות סרגל התקדמות)
    print("Starting processing...")
    for sentence in tqdm(sentences):
        try:
            # הרצת המודל. output_style='json' נותן לנו את המידע הכי נגיש
            prediction = model.predict([sentence], tokenizer, output_style='json')
            
            # 4. חילוץ המידע
            entities = extract_entities(prediction)
            all_results.extend(entities)
            
        except Exception as e:
            print(f"Error processing sentence: {sentence[:30]}... Error: {e}")

    # 5. שמירה לאקסל
    if all_results:
        print(f"Saving {len(all_results)} entities to {OUTPUT_FILE}...")
        df = pd.DataFrame(all_results)
        df.to_excel(OUTPUT_FILE, index=False)
        print("Done! Check the Excel file.")
    else:
        print("No entities found in the text.")

if __name__ == "__main__":
    main()