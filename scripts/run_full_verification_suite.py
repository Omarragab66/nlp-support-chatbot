import sys, os, json
sys.path.insert(0, os.path.abspath("."))
from src.pipeline import CustomerSupportPipeline

bot = CustomerSupportPipeline()

test_suite = [
    # 1. Negative / Escalation
    {"cat": "1. Negative Escalation", "lang": "AR", "query": "الطلب وصلني مكسور ومتبهدل وبقالي أسبوع مستني ومحدش بيرد، ده أسوأ تعامل شفته وعايز فلوسي دلوقتي!"},
    {"cat": "1. Negative Escalation", "lang": "AR", "query": "أنا غاضب جداً، طلبت أوردر بقاله 10 أيام وموصلش وخدمة العملاء سيئة للغاية، فين حاجتي؟"},
    {"cat": "1. Negative Escalation", "lang": "EN", "query": "My package arrived completely damaged and I've been waiting for a response for days, I'm really upset and demand a refund!"},
    {"cat": "1. Negative Escalation", "lang": "EN", "query": "This is terrible service! You charged my card twice for the same order and nobody is helping me, I am extremely furious!"},
    
    # 2. Order Status (Neutral)
    {"cat": "2. Order Status", "lang": "AR", "query": "عايز أعرف شحنتي رقم #54321 هتوصل إمتى وفين مكانها دلوقتي؟"},
    {"cat": "2. Order Status", "lang": "AR", "query": "إزاي أقدر أتبع حالة الأوردر بتاعي على الموقع؟"},
    {"cat": "2. Order Status", "lang": "EN", "query": "How can I track the delivery status of my order #55231?"},
    {"cat": "2. Order Status", "lang": "EN", "query": "What are the available delivery options and shipping times for my order?"},

    # 3. Billing & Refunds (Neutral / Polite)
    {"cat": "3. Billing & Refunds", "lang": "AR", "query": "ما هي سياسة استرجاع المنتجات واسترداد الفلوس عندكم؟"},
    {"cat": "3. Billing & Refunds", "lang": "AR", "query": "عايز أحصل على نسخة من فاتورة الشراء لطلبي الأخير."},
    {"cat": "3. Billing & Refunds", "lang": "EN", "query": "What is your refund policy and how long does it take to process?"},
    {"cat": "3. Billing & Refunds", "lang": "EN", "query": "Can I get an invoice for my last purchase?"},

    # 4. Account Management
    {"cat": "4. Account Management", "lang": "AR", "query": "نسيت كلمة السر بتاعة حسابي، إزاي أقدر أسترجعها؟"},
    {"cat": "4. Account Management", "lang": "AR", "query": "عايز أعدل بياناتي الشخصية على الحساب."},
    {"cat": "4. Account Management", "lang": "EN", "query": "I forgot my account password, how can I reset it?"},
    {"cat": "4. Account Management", "lang": "EN", "query": "I want to edit and update my personal profile information on my account."},

    # 5. Greetings & Small Talk
    {"cat": "5. Small Talk", "lang": "AR", "query": "السلام عليكم، صباح الخير!"},
    {"cat": "5. Small Talk", "lang": "AR", "query": "أهلاً، إزيك عامل إيه؟"},
    {"cat": "5. Small Talk", "lang": "EN", "query": "Hello, good morning! How are you today?"},
    {"cat": "5. Small Talk", "lang": "EN", "query": "Hi there, can you help me?"},

    # 6. Positive Sentiment
    {"cat": "6. Positive Sentiment", "lang": "AR", "query": "شكراً جزيلاً، الخدمة كانت ممتازة وسريعة جداً وأنا مبسوط من التعامل معاكم!"},
    {"cat": "6. Positive Sentiment", "lang": "EN", "query": "Thank you so much, the delivery was super fast and everything is perfect, I love shopping here!"},

    # 7. Out of Scope
    {"cat": "7. Out of Scope", "lang": "AR", "query": "اشرحلي إيه هي الثقوب السوداء ونظرية النسبية؟"},
    {"cat": "7. Out of Scope", "lang": "AR", "query": "مين فاز بكأس العالم 2022؟"},
    {"cat": "7. Out of Scope", "lang": "EN", "query": "Can you write a Python script to scrape a website?"},
    {"cat": "7. Out of Scope", "lang": "EN", "query": "What is the distance between the Earth and the Moon?"}
]

print(f'Running Comprehensive Verification Suite on {len(test_suite)} Test Cases...\n')

results = []
for i, item in enumerate(test_suite, 1):
    res = bot.process_message(item['query'])
    summary = {
        'id': i,
        'category': item['cat'],
        'lang_orig': item['lang'],
        'query': item['query'],
        'detected_lang': res['detected_language'],
        'sentiment': res['detected_sentiment'],
        'sentiment_conf': round(res['sentiment_confidence'], 3),
        'intent': res['detected_intent'],
        'intent_conf': round(res['intent_confidence'], 3),
        'routing': res['routing_action'],
        'escalated': res['priority_escalation'],
        'chunks_count': res['retrieved_chunks_count'],
        'response': res['response']
    }
    results.append(summary)
    print(f\"[{i}/{len(test_suite)}] {item['cat']} ({item['lang']}): Intent={res['detected_intent']} | Sent={res['detected_sentiment']} | Esc={res['priority_escalation']} | Action={res['routing_action']}\")

with open('scripts/full_suite_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('\nFull Test Suite Completed! Results written to scripts/full_suite_results.json')
