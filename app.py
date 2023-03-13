
import gradio as gr
#import torch
#import whisper
#from datetime import datetime
from PIL import Image
#import flag
import os
#MY_SECRET_TOKEN=os.environ.get('HF_TOKEN_SD')

#from diffusers import StableDiffusionPipeline

whisper = gr.Blocks.load(name="spaces/sanchit-gandhi/whisper-large-v2")
stable_diffusion = gr.Blocks.load(name="spaces/stabilityai/stable-diffusion")
### ————————————————————————————————————————

title="Whisper to Stable Diffusion"

### ————————————————————————————————————————

#whisper_model = whisper.load_model("small")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_SECRET_TOKEN)
#pipe.to(device)

### ————————————————————————————————————————

def get_images(prompt):
    gallery_dir = stable_diffusion(prompt,"", 9, fn_index=2)
    return [os.path.join(gallery_dir, img) for img in os.listdir(gallery_dir)]


def magic_whisper_to_sd(audio, guidance_scale, nb_iterations, seed):
    
    whisper_results = translate_better(audio)
    prompt = whisper_results[1]
    images = get_images(prompt)
    
    return whisper_results[0], whisper_results[1], images
    
#def diffuse(prompt, guidance_scale, nb_iterations, seed):
#    
#    generator = torch.Generator(device=device).manual_seed(int(seed))
#    
#    print("""
#    —
#    Sending prompt to Stable Diffusion ... 
#    —
#    """)
#    print("prompt: " + prompt)
#    print("guidance scale: " + str(guidance_scale))
#    print("inference steps: " + str(nb_iterations))
#    print("seed: " + str(seed))
#    
#    images_list = pipe(
#            [prompt] * 2, 
#            guidance_scale=guidance_scale,
#            num_inference_steps=nb_iterations, 
#            generator=generator
#        )
#    
#    images = []
#    
#    safe_image = Image.open(r"unsafe.png")
#    
#    for i, image in enumerate(images_list["sample"]):
#        if(images_list["nsfw_content_detected"][i]):
#            images.append(safe_image)
#        else:
#            images.append(image)
#
#    
#    print("Stable Diffusion has finished")
#    print("———————————————————————————————————————————")
#    
#    return images

def translate_better(audio):
    print("""
    —
    Sending audio to Whisper ...
    —
    """)
    transcribe_text_result = whisper(audio, None, "transcribe", api_name="predict")
    translate_text_result = whisper(audio, None, "translate", api_name="predict")
    print("transcript: " + transcribe_text_result)
    print("———————————————————————————————————————————")  
    print("translated: " + translate_text_result)

    return transcribe_text_result, translate_text_result


#def translate(audio):
#    print("""
#    —
#    Sending audio to Whisper ...
#    —
#    """)
#    # current dateTime
#    now = datetime.now()    
#    # convert to string
#    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
#    print('DateTime String:', date_time_str)
#    
#    audio = whisper.load_audio(audio)
#    audio = whisper.pad_or_trim(audio)
#    
#    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
#    
#    _, probs = whisper_model.detect_language(mel)
#    
#    transcript_options = whisper.DecodingOptions(task="transcribe", fp16 = False)
#    translate_options = whisper.DecodingOptions(task="translate", fp16 = False)
#    
#    transcription = whisper.decode(whisper_model, mel, transcript_options)
#    translation = whisper.decode(whisper_model, mel, translate_options)
#    
#    print("language spoken: " + transcription.language)
#    print("transcript: " + transcription.text)
#    print("———————————————————————————————————————————")  
#    print("translated: " + translation.text)
#    if transcription.language == "en":
#        tr_flag = flag.flag('GB')
#    else:
#        tr_flag = flag.flag(transcription.language)    
#    return tr_flag, transcription.text, translation.text



### ————————————————————————————————————————
 
with gr.Blocks(css="style.css") as demo:
    with gr.Column():
        gr.HTML('''
            <h1>
                Whisper to Stable Diffusion
            </h1>
            <p style='text-align: center;'>
                Ask stable diffusion for images by speaking (or singing 🤗) in your native language ! Try it in French 😉
            </p>
            
            <p style='text-align: center;'>
                This demo is wired to the official SD Space • Offered by Sylvain <a href='https://twitter.com/fffiloni' target='_blank'>@fffiloni</a> • <img id='visitor-badge' alt='visitor badge' src='https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.whisper-to-stable-diffusion' style='display: inline-block' /><br />
                —         
            </p>
    
        ''')
#        with gr.Row(elem_id="w2sd_container"):
#            with gr.Column():
            
        gr.Markdown(
            """
             
            ## 1. Record audio or Upload an audio file:
            """
        )
        
        with gr.Tab(label="Record audio input", elem_id="record_tab"):
            with gr.Column():
                record_input = gr.Audio(
                                    source="microphone",
                                    type="filepath", 
                                    show_label=False,
                                    elem_id="record_btn"
                                )
                with gr.Row():
                    audio_r_translate = gr.Button("Check Whisper first ? 👍", elem_id="check_btn_1")              
                    audio_r_direct_sd = gr.Button("Magic Whisper › SD right now!", elem_id="magic_btn_1")
        
        with gr.Tab(label="Upload audio input", elem_id="upload_tab"):
            with gr.Column():
                upload_input = gr.Audio(
                                    source="upload",
                                    type="filepath",
                                    show_label=False,
                                    elem_id="upload_area"
                                )
                with gr.Row():
                    audio_u_translate = gr.Button("Check Whisper first ? 👍", elem_id="check_btn_2")              
                    audio_u_direct_sd = gr.Button("Magic Whisper › SD right now!", elem_id="magic_btn_2")
        
        with gr.Accordion(label="Stable Diffusion Settings", elem_id="sd_settings", visible=False):
            with gr.Row():
                guidance_scale = gr.Slider(2, 15, value = 7, label = 'Guidance Scale')
                nb_iterations = gr.Slider(10, 50, value = 25, step = 1, label = 'Steps')
                seed = gr.Slider(label = "Seed", minimum = 0, maximum = 2147483647, step = 1, randomize = True)
        
        gr.Markdown(
            """
            ## 2. Check Whisper output, correct it if necessary:
            """
        )
        
        with gr.Row():
            
            transcripted_output = gr.Textbox(
                                    label="Transcription in your detected spoken language", 
                                    lines=3,
                                    elem_id="transcripted"
                                )
            #language_detected_output = gr.Textbox(label="Native language", elem_id="spoken_lang",lines=3)
            
        with gr.Column():
            translated_output = gr.Textbox(
                                    label="Transcript translated in English by Whisper", 
                                    lines=4,
                                    elem_id="translated"
                                )
            with gr.Row():
                clear_btn = gr.Button(value="Clear")
                diffuse_btn = gr.Button(value="OK, Diffuse this prompt !", elem_id="diffuse_btn")
                
                clear_btn.click(fn=lambda value: gr.update(value=""), inputs=clear_btn, outputs=translated_output)
    
                
                
                
                    
#            with gr.Column():
                
                
                    
        gr.Markdown("""
            ## 3. Wait for Stable Diffusion Results ☕️
            Inference time is about ~10 seconds, when it's your turn 😬
            """
            ) 
        
        sd_output = gr.Gallery().style(grid=2, height="auto")
                
                
        gr.Markdown("""
            ### 📌 About the models
            <p style='font-size: 1em;line-height: 1.5em;'>   
            <strong>Whisper</strong> is a general-purpose speech recognition model.<br /><br />
            It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. <br />
            —
            </p>
            <p style='font-size: 1em;line-height: 1.5em;'>
            <strong>Stable Diffusion</strong> is a state of the art text-to-image model that generates images from text.
            </p>
            <div id="notice">
                <div>
                LICENSE 
                <p style='font-size: 0.8em;'> 
                The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank">CreativeML Open RAIL-M</a> license.</p>
                <p style='font-size: 0.8em;'>  
                The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license.</p>
                <p style='font-size: 0.8em;'>  
                The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups.</p>
                <p style='font-size: 0.8em;'>  
                For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" target="_blank">read the license</a>.
                 </p>
                 </div>
                 <div>
                 Biases and content acknowledgment
                 <p style='font-size: 0.8em;'>
                 Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence.</p>
                <p style='font-size: 0.8em;'>  
                The model was trained on the <a href="https://laion.ai/blog/laion-5b/" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes.</p>
                <p style='font-size: 0.8em;'>  You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" target="_blank">model card</a>.
                 </p>
                 </div>
             </div>
    
        """, elem_id="about")
                
        audio_r_translate.click(translate_better, 
                                inputs = record_input, 
                                outputs = [
                                    #language_detected_output,
                                    transcripted_output,
                                    translated_output
                                ])
        
        audio_u_translate.click(translate_better,
                                inputs = upload_input, 
                                outputs = [
                                    #language_detected_output,
                                    transcripted_output,
                                    translated_output
                                ]) 
        
        audio_r_direct_sd.click(magic_whisper_to_sd, 
                                inputs = [
                                    record_input, 
                                    guidance_scale, 
                                    nb_iterations, 
                                    seed
                                ], 
                                outputs = [
                                    #language_detected_output,
                                    transcripted_output,
                                    translated_output,
                                    sd_output
                                ])
        
        audio_u_direct_sd.click(magic_whisper_to_sd, 
                                inputs = [
                                    upload_input,
                                    guidance_scale,
                                    nb_iterations,
                                    seed
                                ], 
                                outputs = [
                                    #language_detected_output,
                                    transcripted_output,
                                    translated_output,
                                    sd_output
                                ])
        
        diffuse_btn.click(get_images, 
                              inputs = [
                                  translated_output
                                  ], 
                              outputs = sd_output
                          )
        gr.HTML('''
                <div class="footer">
                    <p>Whisper by <a href="https://github.com/openai/whisper" target="_blank">OpenAI</a> - Stable Diffusion by <a href="https://huggingface.co/CompVis" target="_blank">CompVis</a> and <a href="https://huggingface.co/stabilityai"  target="_blank">Stability AI</a>
                    </p>
                </div>
                ''')
        
    
if __name__ == "__main__":
    demo.queue(max_size=32, concurrency_count=20).launch()
