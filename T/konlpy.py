from konlpy.tag import Okt

okt = Okt()

srntence = "스쉬옴 늬노리가탄 삶."

nouns = okt.nouns(srntence)
phrases = okt.phrases(srntence)
morphs = okt.morphs(srntence)
pos = okt.pos(srntence)

print("명사 추출:",nouns)
print("구 추출:",phrases)
print("형태소 추출:",morphs)
print("품사 태깅깅:",pos)
