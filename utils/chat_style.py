"""Static styles scoped to Streamlit containers; never interpolate user HTML."""

CHAT_CSS = """
<style>
.stMainBlockContainer {max-width: 1040px; padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stBottomBlockContainer"] {max-width: 1040px; margin: auto;}
.st-key-support_header {border-bottom: 1px solid #e6eaf0; padding-bottom: .7rem; margin-bottom: 1rem;}
.st-key-support_header h3 {font-size: 1.25rem; padding: 0;}
.st-key-welcome {padding: 3.5rem 0 1rem;}
.st-key-welcome h2 {font-size: 1.65rem;}
[data-testid="stChatMessage"] {padding: 1rem; border-radius: 14px; margin-bottom: .6rem;}
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {font-size: 1.15rem; line-height: 1.6; padding-top: .5rem;}
[data-testid="stChatMessage"] [data-testid="stImage"] img {max-height: 220px; object-fit: contain; border-radius: 10px;}
[data-testid="stChatMessage"] p {line-height: 1.75;}
[data-testid="stChatInput"] {border-radius: 14px;}
@media (max-width: 640px) {
 .stMainBlockContainer {padding: 1rem .8rem;}
 [data-testid="stBottomBlockContainer"] {padding-left: .8rem; padding-right: .8rem;}
 .st-key-welcome {padding-top: 1rem;}
 .st-key-support_header h3 {font-size: 1.05rem;}
}
</style>
"""
