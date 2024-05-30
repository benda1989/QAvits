htmls = """
<!DOCTYPE html >
<html lang = "en" >
<head >
    <meta charset = "UTF-8" >
    <title > 知识库 < /title >
    <style >
        /* 页面整体样式 * /
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        /* 容器样式，用于居中输入框和按钮 * /
        .container {
            width: 50 %;
            height: 50 %;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .logo-img {
            width: 75 %; / * 根据实际图片大小调整 * /
            height: auto; / * 保持图片比例 * /
            margin-bottom: 10 %; / * 图片与输入框之间的间隔 * /
        }
        /* 输入框样式 * /
        # message {
            width: 50 %; / * 根据需要调整输入框宽度 * /
            padding: 10px;
            font-size: 16px;
        }
        / * 按钮样式 * /
        # send {
            margin-top: 20px; / * 按钮与输入框之间的间隔 * /
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        footer {
            position: absolute; / * 绝对定位，相对于浏览器窗口 * /
            bottom: 10px; / * 距离底部10px * /
            right: 10px; / * 距离右侧10px * /
        }
    < /style >
< / head >
< body >
< div class = "container" >
    < img src = "" alt = "Logo" class = "logo-img" >
< input type = "text" id = "message" placeholder = "写下你的问题" >
< button id = "send" onclick = "sendMessage()" > 解答 < /button >
< / div >
< footer >
    Power by GKK
< /footer >

< script >
    const audioContext = new(window.AudioContext | | window.webkitAudioContext)();
    let audioStarted = false;
    let audioBufferChunks = [];
    let wss = window.location.href.replace("http","ws")
    if(wss.charAt(wss.length-1) === "/"){
        wss+="ws"
    }else{
        wss+="/ws"
    }
    console.log(wss)
    // WebSocket连接
    var ws = new WebSocket(wss);
    // 接收WebSocket消息
    ws.onmessage = function(event) {
        var reader = new FileReader();
        reader.readAsArrayBuffer(event.data);
        reader.onload = function() {
            audioContext.decodeAudioData(this.result, (buffer)=> {
                audioBufferChunks.push(buffer)
                // 如果还没有开始播放，触发第一次播放
                if (!audioStarted) {
                    playSound()
                }
            })
        }
    };

    // 发送消息到WebSocket
    function sendMessage() {
        var message = document.getElementById('message').value;
        ws.send(message);
        document.getElementById("send").disabled = true;
        document.getElementById("send").textContent = "解答中...";
        setTimeout(()=> {
            if (document.getElementById("send").disabled & & !audioStarted){
                explain();
            }
        }, 10*1000)
    };

    ws.onerror = function(error) {
        console.log('WebSocket Error: ' + error);
        alert("遇到点小问题，请刷新页面使用")
    };
    ws.onopen = function() {
        console.log('WebSocket is connected');
    };
    ws.onclose = function() {
        console.log('WebSocket is closed now');
        alert("服务端有更新，请刷新页面使用")
    };
    // 播放帧
    function playSound() {
        try {
            if (audioBufferChunks.length != = 0) {
                audioStarted = true;
                // 创建当前播放音频帧要用的bufferSource
                const source = audioContext.createBufferSource();
                // 把chunks数组里的第一帧放入source里
                source.buffer = audioBufferChunks[0]
                source.connect(audioContext.destination)
                // 开始播放此帧
                source.start(0)
                // 监听结束事件
                source.addEventListener('ended', ()=> {
                    // 去掉已播放完的帧
                    audioBufferChunks.splice(0, 1);
                    // 回调继续播放
                    playSound();
                })
            } else {
                audioStarted = false;
                explain();
            }
        } catch(e) {
            console.log(e)
            audioStarted = false;
            explain();
        }
    };
    function explain() {
           document.getElementById("send").disabled = false;
           document.getElementById("message").value = "";
            document.getElementById("send").textContent = "解答";
    }
< /script >
< / body >
< / html >

"""
