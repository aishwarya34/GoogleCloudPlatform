<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">

  <title>Google Cloud DataLab</title>
  <link rel="shortcut icon" type="image/x-icon" href="/static/favicon.ico">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <link rel="stylesheet" href="/static/components/codemirror/lib/codemirror.css" />
  <link rel="stylesheet" href="/static/style/style.min.css" type="text/css" />
  <link rel="stylesheet" href="/static/fonts/fonts.css" type="text/css" />
  <link rel="stylesheet" href="/static/style/datalab.css" type="text/css" />
  <link rel="stylesheet" id="themeStylesheet" href="/static/style/custom.css" type="text/css" />

  <script src="/static/components/jquery/jquery.min.js"></script>
</head>
<body class="edit_app"
  data-project=""
  data-base-url="/"
  data-ws-url=""
  data-file-name="model.py"
  data-file-path="datalab/Untitled%20Folder/training-data-analyst/courses/machine_learning/deepdive/03_tensorflow/taxifare/trainer/model.py"
  data-feedback-id=""
  data-version-id="1.2.20190115"
  data-signed-in="true"
  data-user-id="anonymous"
  data-account="860946498807-compute@developer.gserviceaccount.com">
  <div id="app">
    <div id="appBar">
    </div>
    <div id="site">
    <div id="appContent">
      <div id="toolbarArea">
        <div id="mainToolbar">
          <div class="btn-toolbar pull-left">
            <div class="btn-group">
              <button type="button" class="toolbar-btn" data-toggle="dropdown" title="File commands">
                <i class="material-icons">insert_drive_file</i> File
                <span class="caret"></span>
              </button>
              <ul class="dropdown-menu">
                <li id="saveButton"><a href="#">Save</a></li>
                <li id="renameButton"><a href="#">Rename</a></li>
                <li class="divider"></li>
                <li id="downloadButton"><a href="#">Download</a></li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      <div id="contentArea">
        <div id="mainArea">
          <div id="mainContent" class="container fileMainContent">
            <div id="ipython-main-app" style="padding:0px">
              <div id="site">
                <div id="texteditor-backdrop">
                  <div id="texteditor-container" style="height:100%;width:100%"></div>
                  <div id='tooltip' class='ipython_tooltip' style='display:none'></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
  </div>
  <script src="/static/components/es6-promise/promise.min.js"></script>
  <script src="/static/components/requirejs/require.js"></script>
  <script>
    requirejs.config({
      baseUrl: '/static/',
      shim: {
        jqueryui: {
          deps: ["jquery"],
          exports: "$"
        }
      }
    });
    requirejs.config({
       map: {
         '*': {
           'contents': 'services/contents',
         }
       }
    });
    window.datalab = {};

    $("#appBar").load("/static/appbar.html", function() {
        requirejs(['websocket'], (websocket) => {
            requirejs(['edit/js/main.min']);
        });
    });
  </script>
  <script src="//www.gstatic.com/feedback/api.js" async="true" defer="true"></script>
</body>
</html>
