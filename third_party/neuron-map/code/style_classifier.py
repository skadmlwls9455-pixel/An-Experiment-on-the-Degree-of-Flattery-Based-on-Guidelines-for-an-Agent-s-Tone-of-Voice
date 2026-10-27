"""S6 suffix classifier v1.0.0. Same segment splitting/trimming as S3b.

This is a surface-ending heuristic, not a semantic or human tone judgment.
"""
import collections
import re
import unicodedata

VERSION = 's6-endings-1.1.0'
LABELS = ('friend', 'haeyo', 'haera', 'hasipsio', 'aegyo', 'eumseum', 'other_casual')
# Longer / more specific families precede the broad declarative -다 and -요.
PRECEDENCE = ('hasipsio', 'aegyo', 'haeyo', 'eumseum', 'haera', 'friend', 'other_casual')
SUFFIXES = {
    'friend': ('어','아','야','지','네','거든','잖아','해','줘','봐','돼','때','래','게',
               '가','와','서','둬','워','져','려','여','셔','켜','해봐','거야','거지','맞아','자'),
    'haeyo': ('요','세요','죠'),
    'haera': ('니','냐','구나','렴','거라','단다','다','주마','하마','으마'),
    'hasipsio': ('습니다','ㅂ니다','니다','습니까','ㅂ니까','니까','십시오','시오','시죠'),
    'aegyo': ('용','당','지용','거든용','해용','했어용'),
    'eumseum': ('음','함','임','됨','듯','봄','둠','줌','감','옴','듦','붊','삶','앎','섦','뭄','어떰','림','짐','움','넘김','짧아짐'),
    'other_casual': ('대','걸','까','더라','는데','데','구먼','구려','오','소','세'),
}
# S3 noun exclusions retained; extra common nouns prevent obvious suffix collisions.
NOUNS = {'필요','중요','주요','수요','강요','소요','개요','동요','요요','담요','민요',
         '바다','사이다','캐나다','소다','판다','상용','내용','사용','이용','고용','비용',
         '가방','식당','담당','정당','강당','설탕','마당','감','삶','앎','처음','다음','마음','이름','걸음','웃음'}

def segments(text):
    for s in re.split(r'[.!?…\n]+', text):
        if not s.strip():
            continue
        tail = s.strip()
        while tail and (tail[-1].isspace() or tail[-1] in '~^ㅋㅎ' or unicodedata.category(tail[-1])[0] in 'PS'):
            tail = tail[:-1]
        m = re.search(r'([가-힣]+)$', tail)
        yield dict(segment=s, tail=m.group(1) if m else '')

def classify(text):
    counts = dict.fromkeys((*LABELS, 'unknown'), 0)
    tails = []
    for row in segments(text):
        w = row['tail']
        kind = 'unknown'
        if w and w not in NOUNS:
            kind = next((k for k in PRECEDENCE if w.endswith(SUFFIXES[k])), 'unknown')
        counts[kind] += 1
        tails.append(dict(**row, label=kind))
    n = sum(counts.values())
    # Unknown segments are in the denominator and can never win a style label.
    winners = [k for k in LABELS if counts[k] > n * .5 and all(counts[k] > counts[j] for j in LABELS if j != k)]
    return dict(version=VERSION, counts=counts, shares={k:v/max(1,n) for k,v in counts.items()},
                segments=n, dominant=winners[0] if winners else 'unknown', tails=tails)
