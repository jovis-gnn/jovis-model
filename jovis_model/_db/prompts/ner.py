NER_PROMPT_KO = """
넌 <sentence> 속에서 <entity>를 추출하는 named entity recognition 작업을 수행해야 해.
<sentence>는 주로 TV 프로그램에 등장하는 배우가 착용했던 상품에 대한 내용이야.
<sentence>와 그 속에서 등장 가능한 <entity list>가 주어졌을 때 찾을 수 있는 모든 <entity>를 추출 해서 값과 함께 <format>에 따라 답변해주면 돼.
<entity list>는 ,로 구분되어 있고 <format>은 JSON 형태야.

<sentence>
    {}
</sentence>

<entity list>
    {}
</entity list>

<format>
    "추출된 entity 종류 1": "추출된 entity 값 1", "추출된 entity 종류 2": "추출된 entity 값 2", "추출된 entity 종류 3": "추출된 entity 값 3"
</format>

답변은 <format>에 정의된 내용을 JSON 형태로 대답해줘. 다른 대답은 해줄 필요 없어 JSON 결과만 대답 해줘.
"""

NER_PROMPT_EN = """
You are a expert who can do named entity recognition job.
<korean sentence> is our target korean sentence which is the user's input.
The contents of <korean sentence> is mainly about the products worn by actors in TV programs.
You have to do named entity recognition job in <korean sentence>.

I'll give you available entities <entity list>:
<entity list>
    {}
</entity list>

please extract available entities in <korean sentence> and return the <result>.
<result> should follow <format> in JSON format.
<format>
    "sort of extracted entity 1": "extracted entity value 1",
</format>

The <korean sentence> is {}.
Please return the contents of <result> only in JSON format.
The extracted entity must be in the entity list.
I don't need <result> tag and you don't need to do any other reply.
"""
