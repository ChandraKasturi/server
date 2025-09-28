"""
Learning Achievement Service for processing student learning achievements, badges, and streaks.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid

from repositories.mongo_repository import AchievementRepository, HistoryRepository


class LearningAchievementService:
    """Service for handling learning achievement-related operations."""
    
    def __init__(self):
        """Initialize learning achievement service."""
        self.achievement_repo = AchievementRepository()
        self.history_repo = HistoryRepository()
        
        # Learning achievement definitions
        self.achievement_definitions = {
            "curious_questioner": {
                "name": "Curious Questioner",
                "description": "Ask 10 questions in a single learning session",
                "icon": "question-circle",
                "type": "learning",
                "repeatable": True
            },
            "daily_learner": {
                "name": "Daily Learner",
                "description": "Complete learning sessions on 7 consecutive days",
                "icon": "calendar-check",
                "type": "consistency",
                "repeatable": True
            },
            "subject_explorer": {
                "name": "Subject Explorer",
                "description": "Learn about all 5 subjects in a single day",
                "icon": "compass",
                "type": "learning",
                "repeatable": True
            },
            "deep_diver": {
                "name": "Deep Diver",
                "description": "Spend more than 30 minutes in a single learning session",
                "icon": "diving",
                "type": "learning",
                "repeatable": True
            },
            "knowledge_seeker": {
                "name": "Knowledge Seeker",
                "description": "Ask 100 questions across all subjects",
                "icon": "search",
                "type": "milestone",
                "repeatable": False
            },
            "consistent_learner": {
                "name": "Consistent Learner",
                "description": "Maintain a 30-day learning streak",
                "icon": "streak",
                "type": "consistency",
                "repeatable": True
            },
            "subject_master": {
                "name": "Subject Master",
                "description": "Ask 50 questions in a single subject",
                "icon": "graduation-cap",
                "type": "learning",
                "repeatable": True
            },
            "morning_scholar": {
                "name": "Morning Scholar",
                "description": "Complete learning sessions before 12 PM for 7 consecutive days",
                "icon": "sunrise",
                "type": "consistency",
                "repeatable": True
            },
            "night_owl": {
                "name": "Night Owl",
                "description": "Complete learning sessions after 8 PM for 7 consecutive days",
                "icon": "moon",
                "type": "consistency",
                "repeatable": True
            },
            "weekend_warrior": {
                "name": "Weekend Warrior",
                "description": "Complete learning sessions on weekends for 4 consecutive weeks",
                "icon": "weekend",
                "type": "consistency",
                "repeatable": True
            }
        }
        
        # Learning badge tier thresholds
        self.badge_tiers = {
            "subject_engagement": {
                "bronze": {"sessions": 5, "questions": 20, "days": 7},
                "silver": {"sessions": 15, "questions": 50, "days": 14},
                "gold": {"sessions": 30, "questions": 100, "days": 30},
                "platinum": {"sessions": 50, "questions": 200, "days": 45},
                "diamond": {"sessions": 100, "questions": 500, "days": 60}
            },
            "learning_streak": {
                "bronze": {"streak": 3},
                "silver": {"streak": 7},
                "gold": {"streak": 14},
                "platinum": {"streak": 30},
                "diamond": {"streak": 60}
            },
            "question_mastery": {
                "bronze": {"total_questions": 50},
                "silver": {"total_questions": 150},
                "gold": {"total_questions": 300},
                "platinum": {"total_questions": 500},
                "diamond": {"total_questions": 1000}
            }
        }
    
    def process_learning_interaction(self, student_id: str, interaction_data: Dict) -> Dict[str, Any]:
        """
        Process all learning achievements, badges, and streaks after a learning interaction.
        
        Args:
            student_id: ID of the student
            interaction_data: Learning interaction data (question asked, AI response, etc.)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "achievements_earned": [],
            "badges_updated": [],
            "streaks_updated": [],
            "errors": [],
            "achievement_details": [],
            "badge_details": [],
            "streak_details": []
        }
        
        try:
            # Only process if this is a user question (not AI response)
            if not interaction_data.get("is_ai", True):
                # Process learning achievements
                achievement_results, achievement_details = self._process_learning_achievements(
                    student_id, interaction_data
                )
                results["achievements_earned"].extend(achievement_results)
                results["achievement_details"].extend(achievement_details)
                
                # Process badge updates
                badge_results, badge_details = self._process_learning_badge_updates(
                    student_id, interaction_data
                )
                results["badges_updated"].extend(badge_results)
                results["badge_details"].extend(badge_details)
                
                # Process streak updates
                streak_results, streak_details = self._process_learning_streak_updates(
                    student_id, interaction_data
                )
                results["streaks_updated"].extend(streak_results)
                results["streak_details"].extend(streak_details)
            
        except Exception as e:
            results["errors"].append(f"Error processing learning achievements: {str(e)}")
            print(f"Learning achievement processing error for student {student_id}: {str(e)}")
        
        return results
    
    def _process_learning_achievements(self, student_id: str, interaction_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process learning-based achievements."""
        earned_achievements = []
        achievement_details = []
        
        subject = interaction_data.get("subject", "")
        session_id = interaction_data.get("session_id", "")
        interaction_time = interaction_data.get("time", datetime.utcnow())
        
        if isinstance(interaction_time, str):
            interaction_time = datetime.fromisoformat(interaction_time.replace('Z', '+00:00'))
        
        # Get session stats for current session
        session_stats = self._get_session_stats(student_id, session_id)
        
        # Get daily stats
        daily_stats = self._get_daily_learning_stats(student_id, interaction_time.date())
        
        # Get overall stats
        overall_stats = self._get_overall_learning_stats(student_id)
        
        # Curious Questioner - 10 questions in a single session
        if session_stats.get("user_questions", 0) >= 10:
            if self._award_or_update_learning_achievement(student_id, "curious_questioner", interaction_data):
                earned_achievements.append("curious_questioner")
                achievement_details.append({
                    "id": "curious_questioner",
                    "name": "Curious Questioner",
                    "description": "Ask 10 questions in a single learning session",
                    "icon": "question-circle",
                    "type": "learning",
                    "questions_in_session": session_stats.get("user_questions", 0),
                    "is_new": True,
                    "achievement_description": self.achievement_definitions["curious_questioner"]["description"]
                })
        
        # Subject Explorer - learn about all 5 subjects in a day
        if len(daily_stats.get("subjects_covered", [])) >= 5:
            if self._award_or_update_learning_achievement(student_id, "subject_explorer", interaction_data):
                earned_achievements.append("subject_explorer")
                achievement_details.append({
                    "id": "subject_explorer",
                    "name": "Subject Explorer",
                    "description": "Learn about all 5 subjects in a single day",
                    "icon": "compass",
                    "type": "learning",
                    "subjects_covered": daily_stats.get("subjects_covered", []),
                    "is_new": True,
                    "achievement_description": self.achievement_definitions["subject_explorer"]["description"]
                })
        
        # Deep Diver - more than 30 minutes in a session
        session_duration = session_stats.get("duration_minutes", 0)
        if session_duration > 30:
            if self._award_or_update_learning_achievement(student_id, "deep_diver", interaction_data):
                earned_achievements.append("deep_diver")
                achievement_details.append({
                    "id": "deep_diver",
                    "name": "Deep Diver",
                    "description": "Spend more than 30 minutes in a single learning session",
                    "icon": "diving",
                    "type": "learning",
                    "session_duration": session_duration,
                    "is_new": True,
                    "achievement_description": self.achievement_definitions["deep_diver"]["description"]
                })
        
        # Knowledge Seeker - 100 questions across all subjects
        if overall_stats.get("total_questions", 0) >= 100:
            # Check if this achievement hasn't been awarded yet (non-repeatable)
            existing_achievement = self.achievement_repo.get_achievement_by_id(student_id, "knowledge_seeker")
            if not existing_achievement:
                if self._award_or_update_learning_achievement(student_id, "knowledge_seeker", interaction_data):
                    earned_achievements.append("knowledge_seeker")
                    achievement_details.append({
                        "id": "knowledge_seeker",
                        "name": "Knowledge Seeker",
                        "description": "Ask 100 questions across all subjects",
                        "icon": "search",
                        "type": "milestone",
                        "total_questions": overall_stats.get("total_questions", 0),
                        "is_new": True,
                        "achievement_description": self.achievement_definitions["knowledge_seeker"]["description"]
                    })
        
        # Subject Master - 50 questions in a single subject
        subject_questions = overall_stats.get("by_subject", {}).get(subject, 0)
        if subject_questions >= 50:
            achievement_id = f"subject_master_{subject}"
            if self._award_or_update_learning_achievement(student_id, achievement_id, interaction_data):
                earned_achievements.append(achievement_id)
                achievement_details.append({
                    "id": achievement_id,
                    "name": f"{subject.title()} Subject Master",
                    "description": f"Ask 50 questions in {subject}",
                    "icon": "graduation-cap",
                    "type": "learning",
                    "subject": subject,
                    "subject_questions": subject_questions,
                    "is_new": True,
                    "achievement_description": f"Ask 50 questions in {subject}"
                })
        
        # Time-based achievements
        hour = interaction_time.hour
        
        # Check for morning scholar pattern (before 12 PM for 7 consecutive days)
        if hour < 12:
            morning_streak = self._check_time_pattern_streak(student_id, "morning", 7)
            if morning_streak >= 7:
                if self._award_or_update_learning_achievement(student_id, "morning_scholar", interaction_data):
                    earned_achievements.append("morning_scholar")
                    achievement_details.append({
                        "id": "morning_scholar",
                        "name": "Morning Scholar",
                        "description": "Complete learning sessions before 12 PM for 7 consecutive days",
                        "icon": "sunrise",
                        "type": "consistency",
                        "streak_days": morning_streak,
                        "is_new": True,
                        "achievement_description": self.achievement_definitions["morning_scholar"]["description"]
                    })
        
        # Check for night owl pattern (after 8 PM for 7 consecutive days)
        if hour >= 20:
            night_streak = self._check_time_pattern_streak(student_id, "night", 7)
            if night_streak >= 7:
                if self._award_or_update_learning_achievement(student_id, "night_owl", interaction_data):
                    earned_achievements.append("night_owl")
                    achievement_details.append({
                        "id": "night_owl",
                        "name": "Night Owl",
                        "description": "Complete learning sessions after 8 PM for 7 consecutive days",
                        "icon": "moon",
                        "type": "consistency",
                        "streak_days": night_streak,
                        "is_new": True,
                        "achievement_description": self.achievement_definitions["night_owl"]["description"]
                    })
        
        return earned_achievements, achievement_details
    
    def _process_learning_badge_updates(self, student_id: str, interaction_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process learning badge updates."""
        updated_badges = []
        badge_details = []
        
        subject = interaction_data.get("subject", "")
        
        if not subject:
            return updated_badges, badge_details
        
        # Update Subject Engagement badge
        subject_badge_info = self._update_subject_engagement_badge(student_id, subject)
        if subject_badge_info:
            updated_badges.append(f"subject_engagement_{subject}")
            badge_details.append({
                "id": f"subject_engagement_{subject}",
                "type": "subject_engagement",
                "name": f"{subject.title()} Engagement",
                "subject": subject,
                "tier": subject_badge_info.get("tier"),
                "progress": subject_badge_info.get("progress", {}),
                "is_new_tier": subject_badge_info.get("is_new_tier", False)
            })
        
        # Update Learning Streak badge
        streak_badge_info = self._update_learning_streak_badge(student_id)
        if streak_badge_info:
            updated_badges.append("learning_streak")
            badge_details.append({
                "id": "learning_streak",
                "type": "learning_streak",
                "name": "Learning Streak",
                "tier": streak_badge_info.get("tier"),
                "progress": streak_badge_info.get("progress", {}),
                "is_new_tier": streak_badge_info.get("is_new_tier", False)
            })
        
        # Update Question Mastery badge
        mastery_badge_info = self._update_question_mastery_badge(student_id)
        if mastery_badge_info:
            updated_badges.append("question_mastery")
            badge_details.append({
                "id": "question_mastery",
                "type": "question_mastery",
                "name": "Question Master",
                "tier": mastery_badge_info.get("tier"),
                "progress": mastery_badge_info.get("progress", {}),
                "is_new_tier": mastery_badge_info.get("is_new_tier", False)
            })
        
        return updated_badges, badge_details
    
    def _process_learning_streak_updates(self, student_id: str, interaction_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process learning streak updates."""
        updated_streaks = []
        streak_details = []
        
        interaction_time = interaction_data.get("time", datetime.utcnow())
        if isinstance(interaction_time, str):
            interaction_time = datetime.fromisoformat(interaction_time.replace('Z', '+00:00'))
        
        subject = interaction_data.get("subject", "")
        
        # Update daily learning streak
        daily_streak_info = self._update_daily_learning_streak(student_id, interaction_time)
        if daily_streak_info:
            updated_streaks.append("daily_learning")
            streak_details.append({
                "type": "daily_learning",
                "name": "Daily Learning Streak",
                "current_streak": daily_streak_info.get("current_streak", 0),
                "longest_streak": daily_streak_info.get("longest_streak", 0),
                "is_new_record": daily_streak_info.get("is_new_record", False)
            })
        
        # Update subject learning streak
        if subject:
            subject_streak_info = self._update_subject_learning_streak(student_id, subject, interaction_time)
            if subject_streak_info:
                updated_streaks.append(f"subject_learning_{subject}")
                streak_details.append({
                    "type": "subject_learning",
                    "name": f"{subject.title()} Learning Streak",
                    "subject": subject,
                    "current_streak": subject_streak_info.get("current_streak", 0),
                    "longest_streak": subject_streak_info.get("longest_streak", 0),
                    "is_new_record": subject_streak_info.get("is_new_record", False)
                })
        
        return updated_streaks, streak_details
    
    def _get_session_stats(self, student_id: str, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific learning session."""
        try:
            collection = self.history_repo.get_collection(student_id, "sahasra_history")
            
            # Get all messages in this session
            session_messages = list(collection.find({"session_id": session_id}))
            
            user_questions = len([msg for msg in session_messages if not msg.get("is_ai", True)])
            ai_responses = len([msg for msg in session_messages if msg.get("is_ai", False)])
            
            # Calculate session duration
            if session_messages:
                times = [msg.get("time") for msg in session_messages if msg.get("time")]
                if times:
                    start_time = min(times)
                    end_time = max(times)
                    duration_minutes = (end_time - start_time).total_seconds() / 60
                else:
                    duration_minutes = 0
            else:
                duration_minutes = 0
            
            return {
                "user_questions": user_questions,
                "ai_responses": ai_responses,
                "total_messages": len(session_messages),
                "duration_minutes": duration_minutes
            }
            
        except Exception as e:
            print(f"Error getting session stats: {str(e)}")
            return {"user_questions": 0, "ai_responses": 0, "total_messages": 0, "duration_minutes": 0}
    
    def _get_daily_learning_stats(self, student_id: str, date: datetime.date) -> Dict[str, Any]:
        """Get learning statistics for a specific day."""
        try:
            collection = self.history_repo.get_collection(student_id, "sahasra_history")
            
            # Get start and end of the day
            start_of_day = datetime.combine(date, datetime.min.time())
            end_of_day = datetime.combine(date, datetime.max.time())
            
            # Get all user messages for this day
            daily_messages = list(collection.find({
                "is_ai": False,
                "time": {"$gte": start_of_day, "$lte": end_of_day}
            }))
            
            # Get unique subjects covered
            subjects_covered = list(set([msg.get("subject") for msg in daily_messages if msg.get("subject")]))
            
            # Get unique sessions
            sessions = list(set([msg.get("session_id") for msg in daily_messages if msg.get("session_id")]))
            
            return {
                "total_questions": len(daily_messages),
                "subjects_covered": subjects_covered,
                "sessions_count": len(sessions)
            }
            
        except Exception as e:
            print(f"Error getting daily stats: {str(e)}")
            return {"total_questions": 0, "subjects_covered": [], "sessions_count": 0}
    
    def _get_overall_learning_stats(self, student_id: str) -> Dict[str, Any]:
        """Get overall learning statistics for the student."""
        try:
            # Use the existing method from history repository
            questions_stats = self.history_repo.get_questions_answered_count(student_id)
            
            return {
                "total_questions": questions_stats.get("total_questions_answered", 0),
                "by_subject": questions_stats.get("by_subject", {})
            }
            
        except Exception as e:
            print(f"Error getting overall stats: {str(e)}")
            return {"total_questions": 0, "by_subject": {}}
    
    def _check_time_pattern_streak(self, student_id: str, pattern_type: str, required_days: int) -> int:
        """Check for time-based learning patterns (morning/night)."""
        try:
            collection = self.history_repo.get_collection(student_id, "sahasra_history")
            
            # Get the last required_days worth of data
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=required_days - 1)
            
            consecutive_days = 0
            current_date = end_date
            
            for i in range(required_days):
                check_date = current_date - timedelta(days=i)
                start_of_day = datetime.combine(check_date, datetime.min.time())
                end_of_day = datetime.combine(check_date, datetime.max.time())
                
                # Get user messages for this day
                daily_messages = list(collection.find({
                    "is_ai": False,
                    "time": {"$gte": start_of_day, "$lte": end_of_day}
                }))
                
                # Check if any message matches the pattern
                pattern_match = False
                for msg in daily_messages:
                    hour = msg.get("time", datetime.utcnow()).hour
                    if pattern_type == "morning" and hour < 12:
                        pattern_match = True
                        break
                    elif pattern_type == "night" and hour >= 20:
                        pattern_match = True
                        break
                
                if pattern_match:
                    consecutive_days += 1
                else:
                    break
            
            return consecutive_days
            
        except Exception as e:
            print(f"Error checking time pattern streak: {str(e)}")
            return 0
    
    def _award_or_update_learning_achievement(self, student_id: str, achievement_id: str, interaction_data: Dict) -> bool:
        """Award a new learning achievement or update count for existing one."""
        try:
            # Check if achievement already exists
            existing_achievement = self.achievement_repo.get_achievement_by_id(student_id, achievement_id)
            
            if existing_achievement:
                # Update count for repeatable achievement
                return self.achievement_repo.update_achievement_count(student_id, achievement_id)
            else:
                # Create new achievement
                # Handle custom achievement IDs (like subject_master_science)
                base_achievement_id = achievement_id.split("_")[0] + "_" + achievement_id.split("_")[1]
                if base_achievement_id not in self.achievement_definitions:
                    base_achievement_id = "subject_master"  # fallback
                
                achievement_def = self.achievement_definitions.get(base_achievement_id)
                if not achievement_def:
                    return False
                
                achievement_data = {
                    "achievement_id": achievement_id,
                    "achievement_type": achievement_def["type"],
                    "name": achievement_def["name"],
                    "description": achievement_def["description"],
                    "icon": achievement_def["icon"],
                    "count": 1,
                    "metadata": {
                        "subject": interaction_data.get("subject"),
                        "session_id": interaction_data.get("session_id"),
                        "interaction_time": interaction_data.get("time", datetime.utcnow()).isoformat()
                    }
                }
                
                result = self.achievement_repo.add_achievement(student_id, achievement_data)
                return bool(result)
                
        except Exception as e:
            print(f"Error awarding learning achievement {achievement_id}: {str(e)}")
            return False
    
    def _update_subject_engagement_badge(self, student_id: str, subject: str) -> Optional[Dict]:
        """Update Subject Engagement badge for a subject."""
        try:
            # Get learning statistics for this subject
            stats = self._get_subject_learning_stats(student_id, subject)
            
            sessions = stats.get("sessions", 0)
            questions = stats.get("total_questions", 0)
            days_active = stats.get("days_active", 0)
            
            # Determine appropriate tier
            new_tier = self._calculate_engagement_tier(sessions, questions, days_active)
            
            if not new_tier:
                return None
            
            # Check if this is a new tier
            existing_badge = self.achievement_repo.get_badge_by_id(student_id, f"subject_engagement_{subject}")
            is_new_tier = not existing_badge or existing_badge.get("tier") != new_tier
            
            # Create or update badge
            badge_id = f"subject_engagement_{subject}"
            badge_data = {
                "badge_id": badge_id,
                "badge_type": "subject_engagement",
                "name": f"{subject.title()} Engagement",
                "tier": new_tier,
                "subject": subject,
                "earned_date": datetime.utcnow().isoformat(),
                "progress": {
                    "sessions": sessions,
                    "total_questions": questions,
                    "days_active": days_active
                }
            }
            
            success = self.achievement_repo.upsert_badge(student_id, badge_data)
            if success:
                return {
                    "tier": new_tier,
                    "progress": badge_data["progress"],
                    "is_new_tier": is_new_tier
                }
            return None
            
        except Exception as e:
            print(f"Error updating subject engagement badge for {subject}: {str(e)}")
            return None
    
    def _get_subject_learning_stats(self, student_id: str, subject: str) -> Dict[str, Any]:
        """Get learning statistics for a specific subject."""
        try:
            collection = self.history_repo.get_collection(student_id, "sahasra_history")
            
            # Get all user messages for this subject
            messages = list(collection.find({
                "subject": subject,
                "is_ai": False
            }))
            
            # Count unique sessions
            sessions = len(set([msg.get("session_id") for msg in messages if msg.get("session_id")]))
            
            # Count unique days
            days = set()
            for msg in messages:
                if msg.get("time"):
                    days.add(msg.get("time").date())
            
            return {
                "sessions": sessions,
                "total_questions": len(messages),
                "days_active": len(days)
            }
            
        except Exception as e:
            print(f"Error getting subject learning stats: {str(e)}")
            return {"sessions": 0, "total_questions": 0, "days_active": 0}
    
    def _calculate_engagement_tier(self, sessions: int, questions: int, days: int) -> Optional[str]:
        """Calculate the appropriate tier for subject engagement badge."""
        thresholds = self.badge_tiers["subject_engagement"]
        
        # Check Diamond tier (most restrictive first)
        if (sessions >= thresholds["diamond"]["sessions"] and 
            questions >= thresholds["diamond"]["questions"] and
            days >= thresholds["diamond"]["days"]):
            return "diamond"
        
        # Check Platinum tier
        if (sessions >= thresholds["platinum"]["sessions"] and 
            questions >= thresholds["platinum"]["questions"] and
            days >= thresholds["platinum"]["days"]):
            return "platinum"
        
        # Check Gold tier
        if (sessions >= thresholds["gold"]["sessions"] and 
            questions >= thresholds["gold"]["questions"] and
            days >= thresholds["gold"]["days"]):
            return "gold"
        
        # Check Silver tier
        if (sessions >= thresholds["silver"]["sessions"] and 
            questions >= thresholds["silver"]["questions"] and
            days >= thresholds["silver"]["days"]):
            return "silver"
        
        # Check Bronze tier
        if (sessions >= thresholds["bronze"]["sessions"] and 
            questions >= thresholds["bronze"]["questions"] and
            days >= thresholds["bronze"]["days"]):
            return "bronze"
        
        return None
    
    def _update_learning_streak_badge(self, student_id: str) -> Optional[Dict]:
        """Update Learning Streak badge."""
        try:
            # Get current learning streak
            streak_info = self.history_repo.get_learning_streak(student_id)
            current_streak = streak_info.get("current_streak", 0)
            
            # Determine appropriate tier
            new_tier = self._calculate_streak_tier(current_streak)
            
            if not new_tier:
                return None
            
            # Check if this is a new tier
            existing_badge = self.achievement_repo.get_badge_by_id(student_id, "learning_streak")
            is_new_tier = not existing_badge or existing_badge.get("tier") != new_tier
            
            # Create or update badge
            badge_data = {
                "badge_id": "learning_streak",
                "badge_type": "learning_streak",
                "name": "Learning Streak",
                "tier": new_tier,
                "earned_date": datetime.utcnow().isoformat(),
                "progress": {
                    "current_streak": current_streak,
                    "longest_streak": streak_info.get("longest_streak", 0)
                }
            }
            
            success = self.achievement_repo.upsert_badge(student_id, badge_data)
            if success:
                return {
                    "tier": new_tier,
                    "progress": badge_data["progress"],
                    "is_new_tier": is_new_tier
                }
            return None
            
        except Exception as e:
            print(f"Error updating learning streak badge: {str(e)}")
            return None
    
    def _calculate_streak_tier(self, streak: int) -> Optional[str]:
        """Calculate the appropriate tier for learning streak badge."""
        thresholds = self.badge_tiers["learning_streak"]
        
        if streak >= thresholds["diamond"]["streak"]:
            return "diamond"
        elif streak >= thresholds["platinum"]["streak"]:
            return "platinum"
        elif streak >= thresholds["gold"]["streak"]:
            return "gold"
        elif streak >= thresholds["silver"]["streak"]:
            return "silver"
        elif streak >= thresholds["bronze"]["streak"]:
            return "bronze"
        
        return None
    
    def _update_question_mastery_badge(self, student_id: str) -> Optional[Dict]:
        """Update Question Mastery badge."""
        try:
            # Get overall question statistics
            overall_stats = self._get_overall_learning_stats(student_id)
            total_questions = overall_stats.get("total_questions", 0)
            
            # Determine appropriate tier
            new_tier = self._calculate_mastery_tier(total_questions)
            
            if not new_tier:
                return None
            
            # Check if this is a new tier
            existing_badge = self.achievement_repo.get_badge_by_id(student_id, "question_mastery")
            is_new_tier = not existing_badge or existing_badge.get("tier") != new_tier
            
            # Create or update badge
            badge_data = {
                "badge_id": "question_mastery",
                "badge_type": "question_mastery",
                "name": "Question Master",
                "tier": new_tier,
                "earned_date": datetime.utcnow().isoformat(),
                "progress": {
                    "total_questions": total_questions,
                    "by_subject": overall_stats.get("by_subject", {})
                }
            }
            
            success = self.achievement_repo.upsert_badge(student_id, badge_data)
            if success:
                return {
                    "tier": new_tier,
                    "progress": badge_data["progress"],
                    "is_new_tier": is_new_tier
                }
            return None
            
        except Exception as e:
            print(f"Error updating question mastery badge: {str(e)}")
            return None
    
    def _calculate_mastery_tier(self, total_questions: int) -> Optional[str]:
        """Calculate the appropriate tier for question mastery badge."""
        thresholds = self.badge_tiers["question_mastery"]
        
        if total_questions >= thresholds["diamond"]["total_questions"]:
            return "diamond"
        elif total_questions >= thresholds["platinum"]["total_questions"]:
            return "platinum"
        elif total_questions >= thresholds["gold"]["total_questions"]:
            return "gold"
        elif total_questions >= thresholds["silver"]["total_questions"]:
            return "silver"
        elif total_questions >= thresholds["bronze"]["total_questions"]:
            return "bronze"
        
        return None
    
    def _update_daily_learning_streak(self, student_id: str, interaction_time: datetime) -> Optional[Dict]:
        """Update daily learning streak based on interaction."""
        try:
            # Get current learning streak
            current_streak = self.achievement_repo.get_streak_by_type(student_id, "daily_learning")
            
            today = interaction_time.date()
            
            if not current_streak:
                # Create new streak
                streak_data = {
                    "streak_type": "daily_learning",
                    "current_streak": 1,
                    "longest_streak": 1,
                    "last_activity_date": today.isoformat(),
                    "streak_start_date": today.isoformat()
                }
                success = self.achievement_repo.update_streak(student_id, streak_data)
                if success:
                    return {
                        "current_streak": 1,
                        "longest_streak": 1,
                        "is_new_record": True
                    }
                return None
            
            # Parse last activity date
            last_activity = datetime.fromisoformat(current_streak["last_activity_date"]).date()
            current_count = current_streak.get("current_streak", 0)
            longest = current_streak.get("longest_streak", 0)
            
            # Calculate new streak
            day_diff = (today - last_activity).days
            
            if day_diff == 0:
                # Same day, no change needed
                return None
            elif day_diff == 1:
                # Consecutive day, increment streak
                new_streak = current_count + 1
                new_longest = max(longest, new_streak)
            elif day_diff == 2:
                # 1-day gap, can use grace period
                new_streak = current_count + 1
                new_longest = max(longest, new_streak)
            else:
                # Gap too large, reset streak
                new_streak = 1
                new_longest = longest
            
            is_new_record = new_longest > longest
            
            # Update streak
            streak_data = {
                "streak_type": "daily_learning",
                "current_streak": new_streak,
                "longest_streak": new_longest,
                "last_activity_date": today.isoformat(),
                "streak_start_date": current_streak.get("streak_start_date", today.isoformat())
            }
            
            # Reset start date if streak was reset
            if new_streak == 1 and current_count > 1:
                streak_data["streak_start_date"] = today.isoformat()
            
            success = self.achievement_repo.update_streak(student_id, streak_data)
            if success:
                return {
                    "current_streak": new_streak,
                    "longest_streak": new_longest,
                    "is_new_record": is_new_record
                }
            return None
            
        except Exception as e:
            print(f"Error updating daily learning streak: {str(e)}")
            return None
    
    def _update_subject_learning_streak(self, student_id: str, subject: str, interaction_time: datetime) -> Optional[Dict]:
        """Update subject-specific learning streak."""
        try:
            # Similar logic to daily streak but subject-specific
            current_streak = self.achievement_repo.get_streak_by_type(student_id, "subject_learning", subject)
            
            today = interaction_time.date()
            
            if not current_streak:
                # Create new subject streak
                streak_data = {
                    "streak_type": "subject_learning",
                    "subject": subject,
                    "current_streak": 1,
                    "longest_streak": 1,
                    "last_activity_date": today.isoformat(),
                    "streak_start_date": today.isoformat()
                }
                success = self.achievement_repo.update_streak(student_id, streak_data)
                if success:
                    return {
                        "current_streak": 1,
                        "longest_streak": 1,
                        "is_new_record": True
                    }
                return None
            
            # Parse last activity date
            last_activity = datetime.fromisoformat(current_streak["last_activity_date"]).date()
            current_count = current_streak.get("current_streak", 0)
            longest = current_streak.get("longest_streak", 0)
            
            # Calculate new streak
            day_diff = (today - last_activity).days
            
            if day_diff == 0:
                # Same day, no change needed
                return None
            elif day_diff == 1:
                # Consecutive day, increment streak
                new_streak = current_count + 1
                new_longest = max(longest, new_streak)
            else:
                # Gap, reset streak
                new_streak = 1
                new_longest = longest
            
            is_new_record = new_longest > longest
            
            # Update streak
            streak_data = {
                "streak_type": "subject_learning",
                "subject": subject,
                "current_streak": new_streak,
                "longest_streak": new_longest,
                "last_activity_date": today.isoformat(),
                "streak_start_date": current_streak.get("streak_start_date", today.isoformat())
            }
            
            # Reset start date if streak was reset
            if new_streak == 1 and current_count > 1:
                streak_data["streak_start_date"] = today.isoformat()
            
            success = self.achievement_repo.update_streak(student_id, streak_data)
            if success:
                return {
                    "current_streak": new_streak,
                    "longest_streak": new_longest,
                    "is_new_record": is_new_record
                }
            return None
            
        except Exception as e:
            print(f"Error updating subject learning streak for {subject}: {str(e)}")
            return None
    
    def get_student_learning_achievements(self, student_id: str, achievement_type: str = None) -> Tuple[Dict, int]:
        """Get learning achievements for a student."""
        try:
            achievements = self.achievement_repo.get_student_achievements(student_id, achievement_type)
            
            # Filter for learning-related achievements
            learning_achievements = [
                a for a in achievements 
                if a.get("achievement_type") in ["learning", "consistency", "milestone"]
            ]
            
            # Convert to response format
            formatted_achievements = []
            by_type = {}
            
            for achievement in learning_achievements:
                formatted_achievement = {
                    "achievement_id": achievement["achievement_id"],
                    "achievement_type": achievement["achievement_type"],
                    "name": achievement["name"],
                    "description": achievement["description"],
                    "icon": achievement["icon"],
                    "count": achievement.get("count", 1),
                    "first_earned": achievement["first_earned"].isoformat() if isinstance(achievement["first_earned"], datetime) else achievement["first_earned"],
                    "last_earned": achievement["last_earned"].isoformat() if isinstance(achievement["last_earned"], datetime) else achievement["last_earned"],
                    "metadata": achievement.get("metadata", {})
                }
                formatted_achievements.append(formatted_achievement)
                
                # Count by type
                achievement_type_key = achievement["achievement_type"]
                by_type[achievement_type_key] = by_type.get(achievement_type_key, 0) + 1
            
            return {
                "achievements": formatted_achievements,
                "total_count": len(formatted_achievements),
                "by_type": by_type
            }, 200
            
        except Exception as e:
            return {"message": f"Error getting learning achievements: {str(e)}"}, 500
    
    def get_student_learning_badges(self, student_id: str, subject: str = None) -> Tuple[Dict, int]:
        """Get learning badges for a student."""
        try:
            # Get learning-related badges
            learning_badge_types = ["subject_engagement", "learning_streak", "question_mastery"]
            all_badges = []
            
            for badge_type in learning_badge_types:
                badges = self.achievement_repo.get_student_badges(student_id, badge_type, subject)
                all_badges.extend(badges)
            
            # Convert to response format
            formatted_badges = []
            by_type = {}
            by_tier = {}
            
            for badge in all_badges:
                formatted_badge = {
                    "badge_id": badge["badge_id"],
                    "badge_type": badge["badge_type"],
                    "name": badge["name"],
                    "tier": badge["tier"],
                    "subject": badge.get("subject"),
                    "earned_date": badge["earned_date"],
                    "updated_at": badge["updated_at"],
                    "progress": badge.get("progress", {})
                }
                formatted_badges.append(formatted_badge)
                
                # Count by type and tier
                badge_type_key = badge["badge_type"]
                by_type[badge_type_key] = by_type.get(badge_type_key, 0) + 1
                
                tier = badge["tier"]
                by_tier[tier] = by_tier.get(tier, 0) + 1
            
            return {
                "badges": formatted_badges,
                "total_count": len(formatted_badges),
                "by_type": by_type,
                "by_tier": by_tier
            }, 200
            
        except Exception as e:
            return {"message": f"Error getting learning badges: {str(e)}"}, 500
    
    def get_student_learning_streaks(self, student_id: str) -> Tuple[Dict, int]:
        """Get learning streaks for a student."""
        try:
            # Get learning-related streaks
            learning_streak_types = ["daily_learning", "subject_learning"]
            all_streaks = []
            
            for streak_type in learning_streak_types:
                streaks = self.achievement_repo.get_student_streaks(student_id, streak_type)
                all_streaks.extend(streaks)
            
            # Organize streaks
            daily_learning_streak = None
            subject_learning_streaks = []
            
            formatted_streaks = []
            
            for streak in all_streaks:
                formatted_streak = {
                    "streak_type": streak["streak_type"],
                    "subject": streak.get("subject"),
                    "current_streak": streak["current_streak"],
                    "longest_streak": streak["longest_streak"],
                    "last_activity_date": streak.get("last_activity_date"),
                    "streak_start_date": streak.get("streak_start_date"),
                    "updated_at": streak["updated_at"]
                }
                formatted_streaks.append(formatted_streak)
                
                # Categorize streaks
                if streak["streak_type"] == "daily_learning":
                    daily_learning_streak = formatted_streak
                elif streak["streak_type"] == "subject_learning":
                    subject_learning_streaks.append(formatted_streak)
            
            return {
                "streaks": formatted_streaks,
                "daily_learning_streak": daily_learning_streak,
                "subject_learning_streaks": subject_learning_streaks
            }, 200
            
        except Exception as e:
            return {"message": f"Error getting learning streaks: {str(e)}"}, 500
