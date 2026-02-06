"""
================================================================================
CALCOLO ESPLICITO DELLA METRICA AskQE
================================================================================
Autore: Basato sul paper "AskQE: Question Answering as Automatic Evaluation 
        for Machine Translation" (Ki, Duh, Carpuat - ACL 2025 Findings)
        
Paper: https://arxiv.org/pdf/2504.11582

Questo script calcola la metrica AskQE utilizzando i risultati della baseline 
con il modello Qwen3B.

================================================================================
SPIEGAZIONE TEORICA (dal paper originale)
================================================================================

AskQE è un framework per la valutazione automatica della Machine Translation
basato su Question Answering. L'idea centrale è:

"Una traduzione è inaffidabile se le domande chiave sul testo sorgente 
 producono risposte diverse quando derivate dalla sorgente o dalla 
 back-translation del MT output"

Il flusso del framework è:
1. Question Generation (QG): Data una frase sorgente, genera domande che 
   possono essere risposte basandosi sulla frase.
2. Question Answering (QA): Genera risposte per ogni domanda usando due contesti:
   - Source: la frase originale (ground truth)
   - BT (Backtranslation): la traduzione MT ritradotta nella lingua originale
3. Answer Comparison: Calcola la sovrapposizione (overlap) tra le risposte.

Se le risposte sono simili → la traduzione è probabilmente corretta
Se le risposte sono diverse → la traduzione potrebbe contenere errori critici

================================================================================
METRICHE DI CONFRONTO UTILIZZATE
================================================================================

Il paper utilizza diverse metriche per confrontare le risposte:

1. **F1 Score** (word-level): 
   - Misura la sovrapposizione a livello di parole tra predizione e riferimento
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)
   
2. **Exact Match (EM)**: 
   - 1 se le risposte normalizzate sono identiche, 0 altrimenti
   
3. **chrF**: 
   - Character n-gram F-score, robusto per morfologia
   
4. **BLEU**: 
   - Sentence-level BLEU score
   
5. **SBERT Cosine Similarity**: 
   - Similarità semantica usando Sentence-BERT embeddings
   - Permette di catturare similarità di significato anche con parole diverse

================================================================================
INTERPRETAZIONE DEI RISULTATI
================================================================================

Perturbazioni MINOR (errori minori, non cambiano il significato):
- spelling, word_order, synonym, intensifier, expansion_noimpact
- Ci aspettiamo score ALTI (risposte simili perché il significato è preservato)

Perturbazioni CRITICAL (errori critici, cambiano il significato):
- expansion_impact, omission, alteration
- Ci aspettiamo score PIÙ BASSI (risposte diverse perché il significato cambia)

Un buon sistema AskQE dovrebbe discriminare tra errori minor e critical.
================================================================================
"""

import json
import os
from collections import Counter
import string
import re
from typing import List, Dict, Tuple, Union

# ============================================================================
# FUNZIONI DI UTILITÀ PER IL CALCOLO DELLE METRICHE
# ============================================================================

def normalize_answer(s: str) -> str:
    """
    Normalizza il testo per il confronto:
    - Rimuove articoli (a, an, the)
    - Rimuove punteggiatura
    - Normalizza spazi bianchi
    
    Basato su: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def f1_score(prediction: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Calcola l'F1 Score a livello di parole.
    
    Formula dal paper (sezione 4.1):
    - Precision = |common_tokens| / |prediction_tokens|
    - Recall = |common_tokens| / |ground_truth_tokens|  
    - F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    if normalize:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction: str, ground_truth: str, normalize: bool = True) -> float:
    """
    Calcola l'Exact Match Score.
    1.0 se le stringhe normalizzate sono identiche, 0.0 altrimenti.
    """
    if normalize:
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    return 1.0 if prediction == ground_truth else 0.0


# ============================================================================
# CLASSE PRINCIPALE PER IL CALCOLO AskQE
# ============================================================================

class AskQECalculator:
    """
    Calcolatore della metrica AskQE.
    
    Implementa il calcolo della metrica come descritto nel paper:
    "The answer overlap score is computed by comparing the predicted answer
    (from backtranslated MT) with the reference answer (from source)."
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.qa_source_dir = os.path.join(results_dir, "QA", "source")
        self.qa_bt_dir = os.path.join(results_dir, "QA", "bt")
        self.qg_dir = os.path.join(results_dir, "QG")
        
        # Configurazione delle perturbazioni (dal paper, Sezione 3: ContraTICO)
        self.minor_perturbations = ["spelling", "word_order", "synonym", 
                                     "intensifier", "expansion_noimpact"]
        self.critical_perturbations = ["expansion_impact", "omission", "alteration"]
        self.all_perturbations = self.minor_perturbations + self.critical_perturbations
        
    def load_answers(self, filepath: str) -> List[Dict]:
        """Carica le risposte da un file JSONL."""
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
        return data
    
    def parse_answers(self, answers_field) -> List[str]:
        """
        Parsa il campo 'answers' che può essere una stringa JSON o una lista.
        """
        if isinstance(answers_field, str):
            try:
                answers = json.loads(answers_field)
                if isinstance(answers, list):
                    return [str(a) for a in answers if a]
            except json.JSONDecodeError:
                return []
        elif isinstance(answers_field, list):
            return [str(a) for a in answers_field if a]
        return []
    
    def calculate_sentence_metrics(self, 
                                    source_answers: List[str], 
                                    bt_answers: List[str]) -> Dict[str, float]:
        """
        Calcola le metriche di confronto per una singola frase.
        
        Dal paper (Sezione 4.1):
        "We compare the answer to each question based on the source sentence 
        (reference) and the backtranslated MT output (predicted)."
        """
        if not source_answers or not bt_answers:
            return None
            
        if len(source_answers) != len(bt_answers):
            # Considera solo le domande comuni
            min_len = min(len(source_answers), len(bt_answers))
            source_answers = source_answers[:min_len]
            bt_answers = bt_answers[:min_len]
        
        f1_scores = []
        em_scores = []
        
        for src_ans, bt_ans in zip(source_answers, bt_answers):
            if not src_ans.strip() or not bt_ans.strip():
                continue
                
            f1 = f1_score(bt_ans, src_ans)
            em = exact_match_score(bt_ans, src_ans)
            
            f1_scores.append(f1)
            em_scores.append(em)
        
        if not f1_scores:
            return None
            
        return {
            "f1": sum(f1_scores) / len(f1_scores),
            "em": sum(em_scores) / len(em_scores),
            "num_questions": len(f1_scores)
        }
    
    def calculate_askqe_for_perturbation(self, 
                                          language: str, 
                                          pipeline: str, 
                                          perturbation: str) -> Dict:
        """
        Calcola la metrica AskQE per una specifica combinazione 
        language-pipeline-perturbation.
        
        Questa è l'implementazione dell'Equazione 1 del paper:
        AskQE(s, t) = AnswerOverlap(A_s, A_bt)
        
        dove:
        - s = source sentence
        - t = MT output  
        - A_s = risposte basate sulla source
        - A_bt = risposte basate sulla backtranslation
        """
        source_file = os.path.join(self.qa_source_dir, f"en-{pipeline}.jsonl")
        bt_file = os.path.join(self.qa_bt_dir, f"{language}-{pipeline}-{perturbation}.jsonl")
        
        if not os.path.exists(source_file):
            return {"error": f"File sorgente non trovato: {source_file}"}
        if not os.path.exists(bt_file):
            return {"error": f"File BT non trovato: {bt_file}"}
        
        source_data = self.load_answers(source_file)
        bt_data = self.load_answers(bt_file)
        
        all_f1 = []
        all_em = []
        total_questions = 0
        valid_sentences = 0
        
        for src_entry, bt_entry in zip(source_data, bt_data):
            src_answers = self.parse_answers(src_entry.get("answers", []))
            bt_answers = self.parse_answers(bt_entry.get("answers", []))
            
            metrics = self.calculate_sentence_metrics(src_answers, bt_answers)
            
            if metrics:
                all_f1.append(metrics["f1"])
                all_em.append(metrics["em"])
                total_questions += metrics["num_questions"]
                valid_sentences += 1
        
        if not all_f1:
            return {"error": "Nessun confronto valido trovato"}
        
        return {
            "language": language,
            "pipeline": pipeline,
            "perturbation": perturbation,
            "avg_f1": sum(all_f1) / len(all_f1),
            "avg_em": sum(all_em) / len(all_em),
            "total_questions": total_questions,
            "valid_sentences": valid_sentences,
            "is_critical": perturbation in self.critical_perturbations
        }
    
    def calculate_all_metrics(self, 
                               languages: List[str] = None, 
                               pipelines: List[str] = None) -> Dict:
        """
        Calcola tutte le metriche AskQE per le combinazioni specificate.
        """
        if languages is None:
            languages = ["es", "fr"]
        if pipelines is None:
            pipelines = ["atomic"]
        
        results = {
            "by_perturbation": {},
            "by_language": {},
            "summary": {}
        }
        
        all_metrics = []
        
        for lang in languages:
            results["by_language"][lang] = {"minor": [], "critical": []}
            
            for pipeline in pipelines:
                for perturbation in self.all_perturbations:
                    print(f"Calcolando: {lang}-{pipeline}-{perturbation}...")
                    
                    metrics = self.calculate_askqe_for_perturbation(
                        lang, pipeline, perturbation
                    )
                    
                    if "error" not in metrics:
                        all_metrics.append(metrics)
                        
                        key = f"{lang}_{perturbation}"
                        results["by_perturbation"][key] = metrics
                        
                        if metrics["is_critical"]:
                            results["by_language"][lang]["critical"].append(metrics)
                        else:
                            results["by_language"][lang]["minor"].append(metrics)
        
        # Calcolo sommario
        if all_metrics:
            minor_metrics = [m for m in all_metrics if not m["is_critical"]]
            critical_metrics = [m for m in all_metrics if m["is_critical"]]
            
            if minor_metrics:
                results["summary"]["minor"] = {
                    "avg_f1": sum(m["avg_f1"] for m in minor_metrics) / len(minor_metrics),
                    "avg_em": sum(m["avg_em"] for m in minor_metrics) / len(minor_metrics),
                    "count": len(minor_metrics)
                }
            
            if critical_metrics:
                results["summary"]["critical"] = {
                    "avg_f1": sum(m["avg_f1"] for m in critical_metrics) / len(critical_metrics),
                    "avg_em": sum(m["avg_em"] for m in critical_metrics) / len(critical_metrics),
                    "count": len(critical_metrics)
                }
            
            results["summary"]["all"] = {
                "avg_f1": sum(m["avg_f1"] for m in all_metrics) / len(all_metrics),
                "avg_em": sum(m["avg_em"] for m in all_metrics) / len(all_metrics),
                "count": len(all_metrics)
            }
        
        return results


# ============================================================================
# MAIN: ESECUZIONE DEL CALCOLO
# ============================================================================

def main():
    # Trova la directory dei risultati
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = script_dir  # Siamo già nella directory dei risultati
    
    print("=" * 80)
    print("CALCOLO METRICA AskQE - Baseline Qwen3B")
    print("=" * 80)
    print()
    
    # Inizializza il calcolatore
    calculator = AskQECalculator(results_dir)
    
    # Calcola le metriche
    results = calculator.calculate_all_metrics(
        languages=["es", "fr"],
        pipelines=["atomic"]
    )
    
    # Stampa i risultati dettagliati
    print()
    print("=" * 80)
    print("RISULTATI DETTAGLIATI PER PERTURBAZIONE")
    print("=" * 80)
    
    print(f"\n{'Perturbazione':<30} {'Lingua':<8} {'F1':<10} {'EM':<10} {'Tipo':<10}")
    print("-" * 70)
    
    for key, metrics in sorted(results["by_perturbation"].items()):
        perturbation = metrics["perturbation"]
        lang = metrics["language"]
        tipo = "CRITICAL" if metrics["is_critical"] else "minor"
        print(f"{perturbation:<30} {lang:<8} {metrics['avg_f1']:.4f}    {metrics['avg_em']:.4f}    {tipo}")
    
    # Stampa sommario
    print()
    print("=" * 80)
    print("SOMMARIO (dal paper, Tabella 3)")
    print("=" * 80)
    
    if "minor" in results["summary"]:
        minor = results["summary"]["minor"]
        print(f"\nErrori MINOR (non cambiano significato):")
        print(f"  - Average F1:  {minor['avg_f1']:.4f}")
        print(f"  - Average EM:  {minor['avg_em']:.4f}")
        print(f"  - Conteggio:   {minor['count']} combinazioni")
    
    if "critical" in results["summary"]:
        critical = results["summary"]["critical"]
        print(f"\nErrori CRITICAL (cambiano significato):")
        print(f"  - Average F1:  {critical['avg_f1']:.4f}")
        print(f"  - Average EM:  {critical['avg_em']:.4f}")
        print(f"  - Conteggio:   {critical['count']} combinazioni")
    
    if "minor" in results["summary"] and "critical" in results["summary"]:
        delta_f1 = results["summary"]["minor"]["avg_f1"] - results["summary"]["critical"]["avg_f1"]
        delta_em = results["summary"]["minor"]["avg_em"] - results["summary"]["critical"]["avg_em"]
        print(f"\nDIFFERENZA (Minor - Critical):")
        print(f"  - Delta F1:    {delta_f1:+.4f}")
        print(f"  - Delta EM:    {delta_em:+.4f}")
        print()
        print("INTERPRETAZIONE:")
        if delta_f1 > 0:
            print("  ✓ Il modello discrimina correttamente: errori minor hanno score più alti")
            print("    di errori critical, indicando che AskQE rileva gli errori semantici.")
        else:
            print("  ✗ Il modello NON discrimina bene tra errori minor e critical.")
    
    # Salva anche in JSON
    output_file = os.path.join(results_dir, "askqe_metrics_summary.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRisultati salvati in: {output_file}")
    
    print()
    print("=" * 80)
    print("SPIEGAZIONE TEORICA (dal paper)")
    print("=" * 80)
    print("""
La metrica AskQE si basa sull'idea che:

1. QUESTION GENERATION (QG):
   - Genera domande dalla frase sorgente usando fatti atomici estratti
   - Pipeline 'atomic': usa NLI per filtrare fatti entailed dalla sorgente
   
2. QUESTION ANSWERING (QA):
   - Risponde alle domande usando DUE contesti:
     a) SOURCE: la frase originale inglese → risposte di riferimento
     b) BACKTRANSLATION: la traduzione MT ritradotta → risposte predette
   
3. ANSWER OVERLAP:
   - Confronta le risposte usando F1 e Exact Match
   - Risposte simili → traduzione fedele
   - Risposte diverse → errore di traduzione rilevato

La chiave è che errori CRITICAL (omission, alteration, expansion_impact)
cambiano il significato e quindi producono risposte diverse, mentre
errori MINOR (spelling, synonym, word_order) preservano il significato
e producono risposte simili.
""")


if __name__ == "__main__":
    main()
