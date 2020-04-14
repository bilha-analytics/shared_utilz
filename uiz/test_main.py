from flask import Flask, render_template, request, url_for

app = Flask(__name__)

### example messages
chat_messages = [
    {
        'src': 'in',
        'msg': "Hi there",
    },
    {
        'src': 'out',
        'msg': "Hi. How can I help you today",
    },
    {
        'src': 'in',
        'msg': "What is rona?",
    },
    {
        'src': 'out',
        'msg': "The quick brown fox jumped over the lazy dogs. This is a test message going out to you",
    },
    {
        'src': 'in',
        'msg': "How is it sent out and why is it not going away already???!!!!",
    },
    {
        'src': 'out',
        'msg': "Bootstrap supports all the HTML5 input types: text, password, datetime, datetime-local, date, month, time, week, number, email, url, search, tel, and color. Note: Inputs will NOT be fully styled if their type is not properly declared!",
    },
    {
        'src': 'in',
        'msg': "Great!",
    },
    {
        'src': 'out',
        'msg': "Any time.\nAnything else?",
    },
]

# entry
@app.route('/', methods=['GET', 'POST'] )
@app.route('/home', methods=['GET', 'POST'] )
@app.route('/home/<user>', methods=['GET', 'POST'] )
def home(user=None):
    # user = 'test' if user is None else user 
    # return "<H1> We've landed!!! {}</H1>".format(user)
    user_que = request.form.get('askBot')
    print(' user asked: ', user_que )

    if user_que:
        chat_messages.append(
            {'src': 'in', 'msg' : user_que }   
        )
        chat_messages.append(
            {'src': 'out', 'msg' : "_".join( user_que.upper() ) }   
        )

    return render_template('widget_chat_box.html', id='bottomz', msgs=chat_messages) 


@app.route('/jhu_map')
def show_jhu_map():
    return render_template('widget_jhu_map.html') 


@app.route('/about')
def about():
    return render_template('widget_about.html') 


if __name__ == "__main__":
    app.run(debug=True) 