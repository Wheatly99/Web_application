from transformers import pipeline, RobertaTokenizerFast
import streamlit as st

st.title('Самый крупный заголовок')
st.header('Заголовок поменьше')
st.subheader('Самый мелкий заголовок')
st.text('Обычный текст')
st.markdown('''более гибкая работа с текстом - через markdown :sunglasses:
''')
st.latex(r'''или + через + latex + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)''')
st.write('еще', 'для', 'вывода текста', 'есть', 'write()')

tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0', max_len=512)
fill_mask = pipeline("fill-mask", model="blinoff/roberta-base-russian-v0", tokenizer=tokenizer)

text_input = st.text_input(label='Введите предложение, замените одно слово на <mask> '
                                 'и нажмите кнопку, или просто нажмите enter',
                           value='Введите предложение, замените одно слово на <mask> и нажмите кнопку')


def predict():
    result = fill_mask(text_input)
    print(result)
    for variant in result:
        s: str = variant['sequence']
        predicted: str = variant['token_str']
        formated = s.replace(predicted, f' **{predicted.replace(" ","")}**')
        st.markdown(formated)


st.button(label='Чудесная кнопка!', on_click=predict())


