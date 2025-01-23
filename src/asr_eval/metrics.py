import jiwer


def cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference=reference, hypothesis=hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference=reference, hypothesis=hypothesis)



def semdist(row): 
    pass 


def sbert_semdist(row):
    pass

def aligned_semdist(row):
    pass
