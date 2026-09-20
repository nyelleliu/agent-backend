import json


class MemoryManager:
    def __init__(
        self,
        redis_client,
        llm_client,
        message_model,
        summary_model,
    ):
        self.redis_client = redis_client
        self.llm_client = llm_client
        self.message_model = message_model
        self.summary_model = summary_model

    def get_chat_history(self, user_id, db):
        Message = self.message_model
        ConversationSummary = self.summary_model

        cache_key = f"summary:{user_id}"
        cached_data = self.redis_client.get(cache_key)

        if cached_data:
            parsed = json.loads(cached_data)
            summary_text = parsed["summary"]
            covers_up_to = parsed["covers_up_to"]
        else:
            summary_record = db.query(ConversationSummary).filter(
                ConversationSummary.user_id == str(user_id)
            ).first()

            if summary_record:
                summary_text = summary_record.summary_text
                covers_up_to = summary_record.covers_up_to_message_id

                self.redis_client.setex(
                    cache_key,
                    300,
                    json.dumps({
                        "summary": summary_text,
                        "covers_up_to": covers_up_to,
                    })
                )
            else:
                summary_text = None
                covers_up_to = 0

        if summary_text:
            recent_rows = db.query(Message).filter(
                Message.user_id == str(user_id),
                Message.id > covers_up_to
            ).order_by(Message.id.asc()).all()

            chat_history = [
                {
                    "role": "system",
                    "content": f"Conversation summary:\n{summary_text}"
                }
            ]

            chat_history.extend([
                {
                    "role": m.role,
                    "content": m.content
                }
                for m in recent_rows
            ])

            return chat_history

        all_rows = db.query(Message).filter(
            Message.user_id == str(user_id)
        ).order_by(Message.id.asc()).all()

        return [
            {
                "role": m.role,
                "content": m.content
            }
            for m in all_rows
        ]

    def maybe_compress_history(self, user_id, db, threshold=20):
        Message = self.message_model
        ConversationSummary = self.summary_model

        summary_record = db.query(ConversationSummary).filter(
            ConversationSummary.user_id == str(user_id)
        ).first()

        covers_up_to = (
            summary_record.covers_up_to_message_id
            if summary_record
            else 0
        )

        uncovered_rows = db.query(Message).filter(
            Message.user_id == str(user_id),
            Message.id > covers_up_to
        ).order_by(Message.id.asc()).all()

        if len(uncovered_rows) < threshold:
            return

        conversation_text = "\n".join(
            f"{message.role}: {message.content}"
            for message in uncovered_rows
        )

        response = self.llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation concisely, "
                        "preserving key facts and context."
                    ),
                },
                {
                    "role": "user",
                    "content": conversation_text,
                },
            ],
        )

        new_summary_text = response.choices[0].message.content
        new_covers_up_to = uncovered_rows[-1].id

        if summary_record:
            summary_record.summary_text = new_summary_text
            summary_record.covers_up_to_message_id = new_covers_up_to
        else:
            summary_record = ConversationSummary(
                user_id=str(user_id),
                summary_text=new_summary_text,
                covers_up_to_message_id=new_covers_up_to,
            )
            db.add(summary_record)

        db.commit()

        self.redis_client.delete(f"summary:{user_id}")
