"""
Achievement Service for processing student achievements, badges, and streaks.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid

from repositories.mongo_repository import AchievementRepository


class AchievementService:
    """Service for handling achievement-related operations."""
    
    def __init__(self):
        """Initialize achievement service."""
        self.achievement_repo = AchievementRepository()
        
        # Achievement definitions
        self.achievement_definitions = {
            "perfect_score": {
                "name": "Perfect Score",
                "description": "Score 100% in any assessment",
                "icon": "trophy",
                "type": "performance",
                "repeatable": True
            },
            "near_perfect": {
                "name": "Near Perfect",
                "description": "Score 95% or higher in an assessment",
                "icon": "medal",
                "type": "performance",
                "repeatable": True
            },
            "hard_hitter": {
                "name": "Hard Hitter",
                "description": "Score 85% or higher on a Hard difficulty assessment",
                "icon": "shield-plus",
                "type": "performance",
                "repeatable": True
            },
            "accuracy_ace": {
                "name": "Accuracy Ace",
                "description": "Score 90% or higher with high accuracy",
                "icon": "target",
                "type": "performance",
                "repeatable": True
            }
        }
        
        # Badge tier thresholds
        self.badge_tiers = {
            "topic_mastery": {
                "bronze": {"attempts": 2, "avg_score": 75, "medium_required": True},
                "silver": {"attempts": 3, "avg_score": 80, "hard_required": True},
                "gold": {"attempts": 4, "avg_score": 85, "hard_attempts": 2, "hard_avg": 80},
                "platinum": {"attempts": 5, "avg_score": 90, "last_hard_score": 90},
                "diamond": {"maintain_days": 30, "min_score": 85, "revalidation_days": 14}
            }
        }
    
    def process_assessment_completion(self, student_id: str, assessment_data: Dict, submission_data: Dict) -> Dict[str, Any]:
        """
        Process all achievements, badges, and streaks after assessment completion.
        
        Args:
            student_id: ID of the student
            assessment_data: Complete assessment data
            submission_data: Submission result data
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "achievements_earned": [],
            "badges_updated": [],
            "streaks_updated": [],
            "errors": [],
            "achievement_details": [],  # Detailed info for frontend display
            "badge_details": [],        # Detailed badge info
            "streak_details": []        # Detailed streak info
        }
        
        try:
            # Process performance achievements
            performance_results, achievement_details = self._process_performance_achievements(
                student_id, assessment_data, submission_data
            )
            results["achievements_earned"].extend(performance_results)
            results["achievement_details"].extend(achievement_details)
            
            # Process badge updates
            badge_results, badge_details = self._process_badge_updates(
                student_id, assessment_data, submission_data
            )
            results["badges_updated"].extend(badge_results)
            results["badge_details"].extend(badge_details)
            
            # Process streak updates
            streak_results, streak_details = self._process_streak_updates(
                student_id, assessment_data, submission_data
            )
            results["streaks_updated"].extend(streak_results)
            results["streak_details"].extend(streak_details)
            
        except Exception as e:
            results["errors"].append(f"Error processing achievements: {str(e)}")
            print(f"Achievement processing error for student {student_id}: {str(e)}")
        
        return results
    
    def _process_performance_achievements(self, student_id: str, assessment_data: Dict, submission_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process performance-based achievements."""
        earned_achievements = []
        achievement_details = []
        score = submission_data.get("score_percentage", 0)
        level = assessment_data.get("level", 1)
        
        # Perfect Score Achievement
        if score == 100:
            if self._award_or_update_achievement(student_id, "perfect_score", assessment_data):
                earned_achievements.append("perfect_score")
                achievement_details.append({
                    "id": "perfect_score",
                    "name": "Perfect Score",
                    "description": "Score 100% in any assessment",
                    "icon": "trophy",
                    "type": "performance",
                    "score_achieved": score,
                    "is_new": True
                })
        
        # Near Perfect Achievement
        elif score >= 95:
            if self._award_or_update_achievement(student_id, "near_perfect", assessment_data):
                earned_achievements.append("near_perfect")
                achievement_details.append({
                    "id": "near_perfect",
                    "name": "Near Perfect",
                    "description": "Score 95% or higher in an assessment",
                    "icon": "medal",
                    "type": "performance",
                    "score_achieved": score,
                    "is_new": True
                })
        
        # Hard Hitter Achievement (Hard difficulty with good score)
        if level >= 3 and score >= 85:
            if self._award_or_update_achievement(student_id, "hard_hitter", assessment_data):
                earned_achievements.append("hard_hitter")
                achievement_details.append({
                    "id": "hard_hitter",
                    "name": "Hard Hitter",
                    "description": "Score 85% or higher on a Hard difficulty assessment",
                    "icon": "shield-plus",
                    "type": "performance",
                    "score_achieved": score,
                    "difficulty_level": level,
                    "is_new": True
                })
        
        # Accuracy Ace Achievement (high accuracy)
        if score >= 90:
            if self._award_or_update_achievement(student_id, "accuracy_ace", assessment_data):
                earned_achievements.append("accuracy_ace")
                achievement_details.append({
                    "id": "accuracy_ace",
                    "name": "Accuracy Ace",
                    "description": "Score 90% or higher with high accuracy",
                    "icon": "target",
                    "type": "performance",
                    "score_achieved": score,
                    "is_new": True
                })
        
        return earned_achievements, achievement_details
    
    def _process_badge_updates(self, student_id: str, assessment_data: Dict, submission_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process badge updates based on assessment completion."""
        updated_badges = []
        badge_details = []
        
        subject = assessment_data.get("subject")
        topics = assessment_data.get("topics", [])
        
        if not subject:
            return updated_badges, badge_details
        
        # Update Topic Mastery badges for each topic
        for topic in topics:
            badge_info = self._update_topic_mastery_badge(student_id, subject, topic)
            if badge_info:
                updated_badges.append(f"topic_mastery_{topic}_{subject}")
                badge_details.append({
                    "id": f"topic_mastery_{topic}_{subject}",
                    "type": "topic_mastery",
                    "name": f"{topic.title()} Master",
                    "subject": subject,
                    "topic": topic,
                    "tier": badge_info.get("tier"),
                    "progress": badge_info.get("progress", {}),
                    "is_new_tier": badge_info.get("is_new_tier", False)
                })
        
        # Update Subject Mastery badge
        subject_badge_info = self._update_subject_mastery_badge(student_id, subject)
        if subject_badge_info:
            updated_badges.append(f"subject_mastery_{subject}")
            badge_details.append({
                "id": f"subject_mastery_{subject}",
                "type": "subject_mastery",
                "name": f"{subject.title()} Expert",
                "subject": subject,
                "topic": None,
                "tier": subject_badge_info.get("tier"),
                "progress": subject_badge_info.get("progress", {}),
                "is_new_tier": subject_badge_info.get("is_new_tier", False)
            })
        
        return updated_badges, badge_details
    
    def _process_streak_updates(self, student_id: str, assessment_data: Dict, submission_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Process streak updates based on assessment completion."""
        updated_streaks = []
        streak_details = []
        
        submission_time = submission_data.get("submission_time", datetime.utcnow())
        if isinstance(submission_time, str):
            submission_time = datetime.fromisoformat(submission_time.replace('Z', '+00:00'))
        
        subject = assessment_data.get("subject")
        
        # Update daily streak
        daily_streak_info = self._update_daily_streak(student_id, submission_time)
        if daily_streak_info:
            updated_streaks.append("daily")
            streak_details.append({
                "type": "daily",
                "name": "Daily Streak",
                "current_streak": daily_streak_info.get("current_streak", 0),
                "longest_streak": daily_streak_info.get("longest_streak", 0),
                "is_new_record": daily_streak_info.get("is_new_record", False)
            })
        
        # Update subject streak
        if subject:
            subject_streak_info = self._update_subject_streak(student_id, subject, submission_time)
            if subject_streak_info:
                updated_streaks.append(f"subject_{subject}")
                streak_details.append({
                    "type": "subject",
                    "name": f"{subject.title()} Streak",
                    "subject": subject,
                    "current_streak": subject_streak_info.get("current_streak", 0),
                    "longest_streak": subject_streak_info.get("longest_streak", 0),
                    "is_new_record": subject_streak_info.get("is_new_record", False)
                })
        
        return updated_streaks, streak_details
    
    def _award_or_update_achievement(self, student_id: str, achievement_id: str, assessment_data: Dict) -> bool:
        """Award a new achievement or update count for existing one."""
        try:
            # Check if achievement already exists
            existing_achievement = self.achievement_repo.get_achievement_by_id(student_id, achievement_id)
            
            if existing_achievement:
                # Update count for repeatable achievement
                return self.achievement_repo.update_achievement_count(student_id, achievement_id)
            else:
                # Create new achievement
                achievement_def = self.achievement_definitions.get(achievement_id)
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
                        "assessment_id": str(assessment_data.get("_id", "")),
                        "subject": assessment_data.get("subject"),
                        "topics": assessment_data.get("topics", [])
                    }
                }
                
                result = self.achievement_repo.add_achievement(student_id, achievement_data)
                return bool(result)
                
        except Exception as e:
            print(f"Error awarding achievement {achievement_id}: {str(e)}")
            return False
    
    def _update_topic_mastery_badge(self, student_id: str, subject: str, topic: str) -> Optional[Dict]:
        """Update Topic Mastery badge for a specific topic."""
        try:
            # Get assessment history for this topic
            history = self.achievement_repo.get_assessment_history_for_topic(
                student_id, subject, topic, days_back=90
            )
            
            if not history:
                return False
            
            # Calculate current stats
            attempts = len(history)
            scores = [h.get("last_submission", {}).get("score_percentage", 0) for h in history]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Count difficulty levels
            hard_attempts = [h for h in history if h.get("level", 1) >= 3]
            medium_attempts = [h for h in history if h.get("level", 1) >= 2]
            
            # Determine appropriate tier
            new_tier = self._calculate_topic_mastery_tier(
                attempts, avg_score, len(hard_attempts), len(medium_attempts), scores
            )
            
            if not new_tier:
                return None
            
            # Check if this is a new tier (tier upgrade)
            existing_badge = self.achievement_repo.get_badge_by_id(student_id, f"topic_mastery_{topic}_{subject}")
            is_new_tier = not existing_badge or existing_badge.get("tier") != new_tier
            
            # Create or update badge
            badge_id = f"topic_mastery_{topic}_{subject}"
            badge_data = {
                "badge_id": badge_id,
                "badge_type": "topic_mastery",
                "name": f"{topic.title()} Master",
                "tier": new_tier,
                "subject": subject,
                "topic": topic,
                "earned_date": datetime.utcnow().isoformat(),
                "progress": {
                    "attempts": attempts,
                    "avg_score": round(avg_score, 2),
                    "hard_attempts": len(hard_attempts),
                    "medium_attempts": len(medium_attempts),
                    "latest_score": scores[-1] if scores else 0
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
            print(f"Error updating topic mastery badge {topic} for {subject}: {str(e)}")
            return None
    
    def _calculate_topic_mastery_tier(self, attempts: int, avg_score: float, hard_attempts: int, 
                                    medium_attempts: int, scores: List[float]) -> Optional[str]:
        """Calculate the appropriate tier for topic mastery badge."""
        thresholds = self.badge_tiers["topic_mastery"]
        
        # Check Diamond tier (most restrictive first)
        if (attempts >= thresholds["platinum"]["attempts"] and 
            avg_score >= thresholds["platinum"]["avg_score"] and
            hard_attempts >= 1 and scores and scores[-1] >= 90):
            return "diamond"
        
        # Check Platinum tier
        if (attempts >= thresholds["platinum"]["attempts"] and 
            avg_score >= thresholds["platinum"]["avg_score"] and
            hard_attempts >= 1 and scores and scores[-1] >= thresholds["platinum"]["last_hard_score"]):
            return "platinum"
        
        # Check Gold tier
        if (attempts >= thresholds["gold"]["attempts"] and 
            avg_score >= thresholds["gold"]["avg_score"] and
            hard_attempts >= thresholds["gold"]["hard_attempts"]):
            return "gold"
        
        # Check Silver tier
        if (attempts >= thresholds["silver"]["attempts"] and 
            avg_score >= thresholds["silver"]["avg_score"] and
            hard_attempts >= 1):
            return "silver"
        
        # Check Bronze tier
        if (attempts >= thresholds["bronze"]["attempts"] and 
            avg_score >= thresholds["bronze"]["avg_score"] and
            medium_attempts >= 1):
            return "bronze"
        
        return None
    
    def _update_subject_mastery_badge(self, student_id: str, subject: str) -> Optional[Dict]:
        """Update Subject Mastery badge for a subject."""
        try:
            # Get topic coverage for this subject
            coverage = self.achievement_repo.get_subject_topic_coverage(student_id, subject)
            attempted_topics = coverage.get("attempted_topics", [])
            
            if not attempted_topics:
                return None
            
            # For now, create a basic subject badge based on topic coverage
            # This can be enhanced with more sophisticated logic later
            topic_count = len(attempted_topics)
            
            # Determine tier based on topic coverage (simplified logic)
            if topic_count >= 5:
                tier = "gold"
            elif topic_count >= 3:
                tier = "silver"
            elif topic_count >= 1:
                tier = "bronze"
            else:
                return None
            
            # Check if this is a new tier (tier upgrade)
            existing_badge = self.achievement_repo.get_badge_by_id(student_id, f"subject_mastery_{subject}")
            is_new_tier = not existing_badge or existing_badge.get("tier") != tier
            
            badge_id = f"subject_mastery_{subject}"
            badge_data = {
                "badge_id": badge_id,
                "badge_type": "subject_mastery",
                "name": f"{subject.title()} Expert",
                "tier": tier,
                "subject": subject,
                "topic": None,
                "earned_date": datetime.utcnow().isoformat(),
                "progress": {
                    "topics_attempted": topic_count,
                    "attempted_topics": attempted_topics
                }
            }
            
            success = self.achievement_repo.upsert_badge(student_id, badge_data)
            if success:
                return {
                    "tier": tier,
                    "progress": badge_data["progress"],
                    "is_new_tier": is_new_tier
                }
            return None
            
        except Exception as e:
            print(f"Error updating subject mastery badge for {subject}: {str(e)}")
            return None
    
    def _update_daily_streak(self, student_id: str, submission_time: datetime) -> Optional[Dict]:
        """Update daily streak based on assessment submission."""
        try:
            # Get current daily streak
            current_streak = self.achievement_repo.get_streak_by_type(student_id, "daily")
            
            today = submission_time.date()
            
            if not current_streak:
                # Create new streak
                streak_data = {
                    "streak_type": "daily",
                    "current_streak": 1,
                    "longest_streak": 1,
                    "last_activity_date": today.isoformat(),
                    "streak_start_date": today.isoformat(),
                    "grace_days_used": 0
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
                # 1-day gap, can use grace period (simplified logic)
                new_streak = current_count + 1
                new_longest = max(longest, new_streak)
            else:
                # Gap too large, reset streak
                new_streak = 1
                new_longest = longest
            
            is_new_record = new_longest > longest
            
            # Update streak
            streak_data = {
                "streak_type": "daily",
                "current_streak": new_streak,
                "longest_streak": new_longest,
                "last_activity_date": today.isoformat(),
                "streak_start_date": current_streak.get("streak_start_date", today.isoformat()),
                "grace_days_used": current_streak.get("grace_days_used", 0)
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
            print(f"Error updating daily streak: {str(e)}")
            return None
    
    def _update_subject_streak(self, student_id: str, subject: str, submission_time: datetime) -> Optional[Dict]:
        """Update subject-specific streak."""
        try:
            # Similar logic to daily streak but subject-specific
            current_streak = self.achievement_repo.get_streak_by_type(student_id, "subject", subject)
            
            today = submission_time.date()
            
            if not current_streak:
                # Create new subject streak
                streak_data = {
                    "streak_type": "subject",
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
                "streak_type": "subject",
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
            print(f"Error updating subject streak for {subject}: {str(e)}")
            return None
    
    def get_student_achievements(self, student_id: str, achievement_type: str = None) -> Tuple[Dict, int]:
        """Get achievements for a student with summary statistics."""
        try:
            achievements = self.achievement_repo.get_student_achievements(student_id, achievement_type)
            
            # Convert to response format
            formatted_achievements = []
            by_type = {}
            
            for achievement in achievements:
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
            return {"message": f"Error getting achievements: {str(e)}"}, 500
    
    def get_student_badges(self, student_id: str, badge_type: str = None, subject: str = None) -> Tuple[Dict, int]:
        """Get badges for a student with summary statistics."""
        try:
            badges = self.achievement_repo.get_student_badges(student_id, badge_type, subject)
            
            # Convert to response format
            formatted_badges = []
            by_type = {}
            by_tier = {}
            
            for badge in badges:
                formatted_badge = {
                    "badge_id": badge["badge_id"],
                    "badge_type": badge["badge_type"],
                    "name": badge["name"],
                    "tier": badge["tier"],
                    "subject": badge.get("subject"),
                    "topic": badge.get("topic"),
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
            return {"message": f"Error getting badges: {str(e)}"}, 500
    
    def get_student_streaks(self, student_id: str) -> Tuple[Dict, int]:
        """Get streaks for a student with organized response."""
        try:
            streaks = self.achievement_repo.get_student_streaks(student_id)
            
            # Organize streaks by type
            daily_streak = None
            subject_streaks = []
            weekly_streak = None
            
            formatted_streaks = []
            
            for streak in streaks:
                formatted_streak = {
                    "streak_type": streak["streak_type"],
                    "subject": streak.get("subject"),
                    "current_streak": streak["current_streak"],
                    "longest_streak": streak["longest_streak"],
                    "last_activity_date": streak.get("last_activity_date"),
                    "streak_start_date": streak.get("streak_start_date"),
                    "updated_at": streak["updated_at"],
                    "grace_days_used": streak.get("grace_days_used", 0),
                    "weekly_progress": streak.get("weekly_progress")
                }
                formatted_streaks.append(formatted_streak)
                
                # Categorize streaks
                if streak["streak_type"] == "daily":
                    daily_streak = formatted_streak
                elif streak["streak_type"] == "subject":
                    subject_streaks.append(formatted_streak)
                elif streak["streak_type"] == "weekly":
                    weekly_streak = formatted_streak
            
            return {
                "streaks": formatted_streaks,
                "daily_streak": daily_streak,
                "subject_streaks": subject_streaks,
                "weekly_streak": weekly_streak
            }, 200
            
        except Exception as e:
            return {"message": f"Error getting streaks: {str(e)}"}, 500
    
    def get_achievement_summary(self, student_id: str) -> Tuple[Dict, int]:
        """Get a comprehensive achievement summary for a student."""
        try:
            # Get all data
            achievements_result, _ = self.get_student_achievements(student_id)
            badges_result, _ = self.get_student_badges(student_id)
            streaks_result, _ = self.get_student_streaks(student_id)
            
            achievements = achievements_result.get("achievements", [])
            badges = badges_result.get("badges", [])
            daily_streak = streaks_result.get("daily_streak")
            
            # Calculate summary statistics
            total_achievements = len(achievements)
            total_badges = len(badges)
            
            # Find highest badge tier
            tier_priority = {"diamond": 5, "platinum": 4, "gold": 3, "silver": 2, "bronze": 1}
            highest_tier = "none"
            for badge in badges:
                badge_tier = badge.get("tier", "bronze")
                if tier_priority.get(badge_tier, 0) > tier_priority.get(highest_tier, 0):
                    highest_tier = badge_tier
            
            # Current and longest daily streak
            current_daily_streak = daily_streak.get("current_streak", 0) if daily_streak else 0
            longest_daily_streak = daily_streak.get("longest_streak", 0) if daily_streak else 0
            
            # Recent achievements (last 5)
            recent_achievements = sorted(
                achievements, 
                key=lambda x: x["last_earned"], 
                reverse=True
            )[:5]
            
            # Featured badges (highest tier badges)
            featured_badges = sorted(
                badges,
                key=lambda x: tier_priority.get(x.get("tier", "bronze"), 0),
                reverse=True
            )[:5]
            
            summary = {
                "total_achievements": total_achievements,
                "total_badges": total_badges,
                "highest_badge_tier": highest_tier,
                "current_daily_streak": current_daily_streak,
                "longest_daily_streak": longest_daily_streak,
                "recent_achievements": recent_achievements,
                "featured_badges": featured_badges
            }
            
            quick_stats = {
                "performance_achievements": achievements_result.get("by_type", {}).get("performance", 0),
                "mastery_badges": badges_result.get("by_type", {}).get("topic_mastery", 0),
                "active_streaks": len([s for s in streaks_result.get("streaks", []) if s.get("current_streak", 0) > 0])
            }
            
            return {
                "summary": summary,
                "quick_stats": quick_stats
            }, 200
            
        except Exception as e:
            return {"message": f"Error getting achievement summary: {str(e)}"}, 500
