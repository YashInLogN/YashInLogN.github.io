from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether

source = Path('/Users/yashchaudhary/.codex/attachments/e5fe9429-a2e2-463b-bf8c-611137fdcfbc/Pasted text.txt')
out = Path('output/pdf/LUMEN - Beginner-Friendly Code Walkthrough.pdf')
text = source.read_text(encoding='utf-8')

font_regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
font_bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if Path(font_regular).exists() and Path(font_bold).exists():
    pdfmetrics.registerFont(TTFont('Arial', font_regular))
    pdfmetrics.registerFont(TTFont('Arial-Bold', font_bold))
    regular, bold = 'Arial', 'Arial-Bold'
else:
    regular, bold = 'Helvetica', 'Helvetica-Bold'

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='DocTitle', parent=styles['Title'], fontName=bold, fontSize=23, leading=29, textColor=colors.HexColor('#12364A'), alignment=TA_CENTER, spaceAfter=8*mm))
styles.add(ParagraphStyle(name='Intro', parent=styles['BodyText'], fontName=regular, fontSize=10.2, leading=15.3, textColor=colors.HexColor('#344956'), spaceAfter=3*mm))
styles.add(ParagraphStyle(name='Section', parent=styles['Heading1'], fontName=bold, fontSize=16, leading=20, textColor=colors.HexColor('#116466'), spaceBefore=6*mm, spaceAfter=3*mm, keepWithNext=True))
styles.add(ParagraphStyle(name='Subsection', parent=styles['Heading2'], fontName=bold, fontSize=11.3, leading=14, textColor=colors.HexColor('#12364A'), spaceBefore=3.3*mm, spaceAfter=1.2*mm, keepWithNext=True))
styles.add(ParagraphStyle(name='BodyClean', parent=styles['BodyText'], fontName=regular, fontSize=9.4, leading=14, textColor=colors.HexColor('#24333B'), spaceAfter=2.1*mm))
styles.add(ParagraphStyle(name='BulletClean', parent=styles['BodyText'], fontName=regular, fontSize=9.2, leading=13.5, textColor=colors.HexColor('#24333B'), leftIndent=5*mm, firstLineIndent=-3*mm, spaceAfter=1.3*mm, bulletIndent=0))
styles.add(ParagraphStyle(name='NumberClean', parent=styles['BodyText'], fontName=regular, fontSize=9.2, leading=13.5, textColor=colors.HexColor('#24333B'), leftIndent=6*mm, firstLineIndent=-5*mm, spaceAfter=1.2*mm))

sections = {
    'app.py — the web app',
    'lumen/data/synthetic.py — sample courses, students, and scores',
    'lumen/features.py — turning details into model inputs',
    'lumen/model/gbm.py — choosing the model type',
    'lumen/model/train.py — training and saving',
    'lumen/recommend.py — scoring and ranking courses',
    'lumen/explain.py — explanations for predictions',
    'lumen/__init__.py and the other __init__.py files',
    'Two important details about what the code means',
}
subheads = {
    'Imports and setup', 'ensure_pipeline()', 'main()', 'Lists and mappings',
    'courses_dataframe()', '_major_dept(major)', '_prereqs_met(completed, prereqs)',
    '_synthetic_fit(...)', 'generate_students(n, seed=42)', 'build_training_pairs(students, seed=42)',
    'save_default_dataset(data_dir, n_students=200, seed=42)',
    'MAJOR_COLS, FEATURE_NAMES, and HUMAN_LABELS', '_parse_completed(val)',
    'row_to_features(...)', 'dataframe_to_feature_matrix(df)', 'student_course_row(student, course)',
    'create_fit_regressor(random_state=42)', 'load_or_create_data(data_dir)',
    'train_model(pairs, model_path, random_state=42)', 'main()', 'load_model(artifacts_dir)',
    'recommend_courses(...)', '_EXPLAINERS and _get_explainer(model)', 'ExplanationBullet',
    '_direction(shap_val)', '_detail_for_feature(name, value, shap_val)',
    'explain_prediction(model, student, course, top_k=5)', 'format_summary(course_name, score, bullets)',
}

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setStrokeColor(colors.HexColor('#DCE5E8'))
        canvas.setLineWidth(0.6)
        canvas.line(20*mm, h-14*mm, w-20*mm, h-14*mm)
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor('#607780'))
        canvas.drawString(20*mm, h-11*mm, 'LUMEN | Beginner-friendly code walkthrough')
    canvas.setStrokeColor(colors.HexColor('#DCE5E8'))
    canvas.line(20*mm, 14*mm, w-20*mm, 14*mm)
    canvas.setFont(regular, 8)
    canvas.setFillColor(colors.HexColor('#607780'))
    canvas.drawRightString(w-20*mm, 9*mm, f'{doc.page}')
    canvas.restoreState()

def p(txt, style='BodyClean'):
    return Paragraph(escape(txt).replace('\n', '<br/>'), styles[style])

story = [Spacer(1, 10*mm), Paragraph('LUMEN', styles['DocTitle']),
         Paragraph('Beginner-Friendly Code Walkthrough', styles['DocTitle']),
         HRFlowable(width='30%', thickness=2, color=colors.HexColor('#25A18E'), hAlign='CENTER', spaceAfter=7*mm)]
lines = text.splitlines()
for i, raw in enumerate(lines):
    line = raw.strip()
    if not line:
        continue
    if i == 0:
        story.append(Paragraph(escape(line), styles['Intro']))
    elif line in sections:
        story.append(Paragraph(escape(line), styles['Section']))
    elif line in subheads:
        story.append(Paragraph(escape(line), styles['Subsection']))
    elif line[:2] in ('- ', '* '):
        story.append(Paragraph('• ' + escape(line[2:]), styles['BulletClean']))
    elif len(line) > 2 and line[0].isdigit() and line[1:3] == '. ':
        story.append(Paragraph(escape(line), styles['NumberClean']))
    else:
        story.append(Paragraph(escape(line), styles['BodyClean']))

doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=21*mm, bottomMargin=20*mm, title='LUMEN - Beginner-Friendly Code Walkthrough', author='OpenAI')
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(out.resolve())
