import plotly.graph_objects as go

def generate_gauge_chart(probability):
    """
    Creates a premium, professional gauge chart using Plotly, matching the new PMOS color scheme.
    """
    percentage = probability * 100
    
    # Define color scale based on the new palette
    if percentage < 30:
        bar_color = "#FBEFEF"  # Background Sekunder (Very Soft Pink)
    elif percentage < 70:
        bar_color = "#F9DFDF"  # Accent Soft (Soft Pink)
    else:
        bar_color = "#F5AFAF"  # Accent Utama (Muted Coral/Rose)
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Skor Risiko PMOS", 'font': {'size': 16, 'color': '#3F3F46', 'family': 'Outfit'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#3F3F46', 'family': 'Outfit', 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#6B7280", 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#FFFFFF",
            'borderwidth': 1,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 30], 'color': '#FCF8F8'},
                {'range': [30, 70], 'color': '#FBEFEF'},
                {'range': [70, 100], 'color': '#F9DFDF'}
            ],
            'threshold': {
                'line': {'color': "#6B7280", 'width': 2},
                'thickness': 0.75,
                'value': 52 # Threshold from the model
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=200
    )
    
    return fig

def generate_probability_meter(probability):
    """
    Creates a clean horizontal comparison meter using Plotly.
    """
    percentage = probability * 100
    
    fig = go.Figure()
    
    # Background bar
    fig.add_trace(go.Bar(
        y=['PMOS Risk'],
        x=[100],
        orientation='h',
        marker=dict(color='#FBEFEF', line=dict(color='#E5E7EB', width=1)),
        hoverinfo='skip'
    ))
    
    # Fill bar
    bar_color = "#F9DFDF" if percentage < 50 else "#F5AFAF"
    fig.add_trace(go.Bar(
        y=['PMOS Risk'],
        x=[percentage],
        orientation='h',
        marker=dict(color=bar_color),
        hoverinfo='x'
    ))
    
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            tickfont=dict(color='#6B7280')
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=50
    )
    
    return fig
