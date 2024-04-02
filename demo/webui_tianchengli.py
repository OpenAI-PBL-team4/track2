import gradio as gr
from track2.demo.inference.inference import greet

default_text = "Nya haha! You may write something here. But don't expect GPT2 to give you some excellent results. :("

input_custom = gr.Textbox(placeholder=default_text)

output_result = gr.Textbox(label="result")
output_time = gr.Textbox(label="time analysis", scale=2)
output_size = gr.Textbox(label="size analysis")
output_compression = gr.Textbox(label="compression analysis")
perplexity = gr.Textbox(label="perplexity")

table_colomn_quantization = gr.Dropdown([
    #"torch.quint8",
    "torch.qint8",
    #"torch.qint32",
    "torch.float16"],
    value="torch.qint8",
    label="Quant Dtype")
slider_pruning_rate = gr.Slider(minimum=0.05, maximum=0.95, step=0.05, label="pruning_ratio")
slider_pruning_iteration = gr.Slider(minimum=1, maximum=30, step=1, label="pruning_iteration")


demo = gr.Interface(fn=greet, inputs=[input_custom,
                                      "checkbox",
                                      slider_pruning_rate,
                                      slider_pruning_iteration,
                                      "checkbox",
                                      table_colomn_quantization,
                                      "checkbox",
                                      "checkbox",
                                      "checkbox"],
                    outputs=[output_result, perplexity, output_time, output_size, output_compression])
demo.launch(debug=True)


