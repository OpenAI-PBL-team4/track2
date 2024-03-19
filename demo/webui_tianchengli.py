import gradio as gr
from track2.demo.inference.inference import greet

default_text = "Nya haha! You may write something here. But don't expect GPT2 to give you some excellent results. :("

input_custom = gr.Textbox(placeholder=default_text)
output_result = gr.Textbox(label="result")
output_time = gr.Textbox(label="time analysis", scale=2)
output_size = gr.Textbox(label="size analysis")
output_compression = gr.Textbox(label="compression analysis")

table_colomn_quantization = gr.Dropdown([
    #"torch.quint8",
    "torch.qint8",
    #"torch.qint32",
    "torch.float16"],
    value="torch.qint8",
    label="Quant Dtype")


demo = gr.Interface(fn=greet, inputs=[input_custom, "checkbox", "checkbox", table_colomn_quantization ,"checkbox","checkbox"],
                    outputs=[output_result, output_time, output_size, output_compression])
demo.launch(debug=True)


