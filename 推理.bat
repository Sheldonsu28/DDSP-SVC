@echo off
chcp 65001
echo ============================================免责声明====================================================
echo 为避免可能的法律纠纷和道德风险，使用者在使用该整合包前，请务必仔细阅读本条款，继续使用即代表理解并同意该声明，如有异议，请立即停止使用并删除本整合包。
echo.
echo 1. 本项目修改自DDSP-SVC项目(https://github.com/yxlllc/DDSP-SVC)
echo.
echo 2. 本整合包仅为交流学习所用，在使用本整合包时，必须根据知情同意原则取得数据集音声来源的授权许可，并根据授权协议条款规定使用数据集。
echo.
echo 3. 禁止使用该整合包对公众人物、政治人物或其他容易引起争议的人物进行模型训练。使用本整合包产出和传输的信息需符合中国法律、国际公约的规定、符合公序良俗。不将本整合包以及与之相关的服务用作非法用途以及非正当用途。
echo.
echo 4. 禁止将本整合包用于血腥、暴力、性相关、或侵犯他人合法权利的用途。
echo.
echo 5. 任何发布到公共平台的基于DDSP-SVC制作的作品，都必须要明确指明用于变声器转换的输入源歌声、音频，例如：使用他人发布的视频/音频，通过分离的人声作为输入源进行转换的，必须要给出明确的原视频、音乐链接；若使用是自己的人声，或是使用其他歌声合成引擎合成的声音作为输入源进行转换的，也必须加以说明。	
echo.			
echo 因使用者违反上述条款中的任意一条或多条而造成的一切后果，均由使用者本人承担，与整合包作者、项目作者无关，特此声明。
echo =========================================================================================================
echo.
echo 请输入使用的模型步数（例：模型为model_2000.pt就输入2000）
set /p d_step=:
echo 请输入推理步数（例：50）
set /p infer_step=:
echo 请输入t_start，即reflow开始时间（例：0.7，若输入0则为纯扩散，1为纯前级）
set /p t_start=:
echo 请选择使用的采样器（例：0为euler，1为rk4）
set /p method=:
if "%method%"=="0" (set method=euler& echo 使用euler)
if "%method%"=="1" (set method=rk4& echo 使用rk4)
set spkid=1
echo 请输入参考的wav干声文件名，该文件应放入raw文件夹下（例：文件名为test.wav就输入test）
set /p wav=:
echo 请输入音高（例：维持原调为0，支持正负，数字为半音）
set /p key=:
echo 请输入formant（例：维持原调为0，支持正负，数字为半音）
set /p formant=:
echo 请输入音区偏移值（例：不使用为0，3为降3key推理后声码器升回来，负数则相反。）
echo （注：使用pc_nsfhifigan保持共振峰同时变调的功能，+-3内提供高保真变调，可能改变真假声，最终输出音高仍为上一个参数值）
set /p v=:
echo 请输入特征混合
set /p weight=:
echo ================================ 成功！开始处理 ================================
.\.venv\Scripts\python.exe main_reflow.py -i raw\%wav%.wav -m exp\reflow-test-new\model_%d_step%.pt -o results\%wav%_%d_step%_%infer_step%_%t_start%_%method%_rmvpe_%key%_%formant%_spkid_%spkid%_formant_key_v=%v%w=%weight%.wav -k %key% -id %spkid% -method %method% -step %infer_step% -pe rmvpe -ts %t_start% -v %v% -f %formant% -fr %weight%
echo ================================ 处理完毕，若无报错则输出至result文件夹 ================================
pause