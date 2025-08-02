"""
Zero Trust Architecture (ZTA) Engine for Q-ZAP Framework
========================================================

This module implements the Zero Trust Architecture engine that provides
dynamic policy enforcement based on real-time risk assessment and
anomaly detection scores from the HAE model.

Core Principles:
- Never trust, always verify
- Least privilege access
- Continuous verification
- Dynamic policy enforcement

Author: Q-ZAP Research Team  
Date: 2025
License: MIT
"""

import time
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import ipaddress
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class EntityType(Enum):
    """Entity type enumeration."""
    USER = "user"
    DEVICE = "device"
    SERVICE = "service"
    APPLICATION = "application"


class PolicyAction(Enum):
    """Policy enforcement actions."""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    ISOLATE = "isolate"
    MONITOR = "monitor"
    STEP_UP_AUTH = "step_up_auth"


@dataclass
class Entity:
    """Represents an entity in the Zero Trust model."""
    entity_id: str
    entity_type: EntityType
    identity: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[Dict[str, str]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class AccessRequest:
    """Represents an access request."""
    request_id: str
    entity: Entity
    resource: str
    action: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    entity_id: str
    risk_level: RiskLevel
    risk_score: float
    anomaly_score: float
    factors: Dict[str, float]
    timestamp: float
    ttl: float = 300.0  # Time to live in seconds


@dataclass
class PolicyDecision:
    """Policy enforcement decision."""
    request_id: str
    action: PolicyAction
    risk_assessment: RiskAssessment
    policy_rules: List[str]
    enforcement_details: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None


@dataclass
class SecurityEvent:
    """Security event for monitoring and logging."""
    event_id: str
    event_type: str
    entity_id: str
    severity: str
    description: str
    details: Dict[str, Any]
    timestamp: float


class RiskCalculator:
    """Calculates risk scores based on multiple factors."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize risk calculator.
        
        Args:
            weights: Factor weights for risk calculation
        """
        self.weights = weights or {
            'anomaly_score': 0.4,
            'location_risk': 0.2,
            'device_trust': 0.2,
            'time_based': 0.1,
            'behavioral': 0.1
        }
    
    def calculate_risk(
        self,
        entity: Entity,
        anomaly_score: float,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Calculate comprehensive risk score for an entity.
        
        Args:
            entity: Entity to assess
            anomaly_score: Anomaly score from HAE model
            historical_data: Historical behavior data
            
        Returns:
            Risk assessment result
        """
        factors = {}
        
        # Anomaly score factor (from HAE model)
        factors['anomaly_score'] = min(anomaly_score, 1.0)
        
        # Location-based risk
        factors['location_risk'] = self._calculate_location_risk(entity)
        
        # Device trust score
        factors['device_trust'] = self._calculate_device_trust(entity)
        
        # Time-based risk (unusual access times)
        factors['time_based'] = self._calculate_time_based_risk(entity, historical_data)
        
        # Behavioral risk
        factors['behavioral'] = self._calculate_behavioral_risk(entity, historical_data)
        
        # Calculate weighted risk score
        risk_score = sum(
            factors[factor] * self.weights.get(factor, 0.0)
            for factor in factors
        )
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return RiskAssessment(
            entity_id=entity.entity_id,
            risk_level=risk_level,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            factors=factors,
            timestamp=time.time()
        )
    
    def _calculate_location_risk(self, entity: Entity) -> float:
        """Calculate location-based risk score."""
        if not entity.location:
            return 0.3  # Unknown location = medium risk
        
        # Example location risk calculation
        country = entity.location.get('country', '').upper()
        high_risk_countries = {'CN', 'RU', 'KP', 'IR'}  # Example high-risk countries
        
        if country in high_risk_countries:
            return 0.8
        elif entity.location.get('vpn_detected'):
            return 0.6
        elif entity.location.get('tor_detected'):
            return 0.9
        else:
            return 0.1
    
    def _calculate_device_trust(self, entity: Entity) -> float:
        """Calculate device trust score."""
        if not entity.device_id:
            return 0.7  # No device info = higher risk
        
        device_attrs = entity.attributes.get('device', {})
        
        # Check device compliance factors
        risk_factors = 0.0
        
        if not device_attrs.get('managed', False):
            risk_factors += 0.3
        
        if not device_attrs.get('encrypted', False):
            risk_factors += 0.2
        
        if device_attrs.get('jailbroken', False):
            risk_factors += 0.4
        
        if not device_attrs.get('up_to_date', True):
            risk_factors += 0.2
        
        return min(risk_factors, 1.0)
    
    def _calculate_time_based_risk(self, entity: Entity, historical_data: Optional[Dict] = None) -> float:
        """Calculate time-based risk score."""
        current_hour = datetime.now().hour
        
        # Business hours (9 AM - 5 PM) are low risk
        if 9 <= current_hour <= 17:
            return 0.1
        # Evening hours (6 PM - 10 PM) are medium risk
        elif 18 <= current_hour <= 22:
            return 0.3
        # Night hours (11 PM - 8 AM) are high risk
        else:
            return 0.7
    
    def _calculate_behavioral_risk(self, entity: Entity, historical_data: Optional[Dict] = None) -> float:
        """Calculate behavioral risk based on historical patterns."""
        if not historical_data:
            return 0.2  # No history = slight risk
        
        # Example behavioral analysis
        typical_locations = historical_data.get('typical_locations', [])
        current_location = entity.ip_address
        
        if current_location and current_location not in typical_locations:
            return 0.6
        
        return 0.1


class PolicyEngine:
    """Core policy enforcement engine."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize policy engine.
        
        Args:
            config_file: Path to policy configuration file
        """
        self.policies = self._load_policies(config_file)
        self.risk_calculator = RiskCalculator()
        self.decision_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def _load_policies(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load policy configuration."""
        default_policies = {
            "access_policies": [
                {
                    "name": "high_risk_deny",
                    "condition": "risk_level == 'critical'",
                    "action": "deny",
                    "priority": 1
                },
                {
                    "name": "high_risk_isolate", 
                    "condition": "risk_level == 'high'",
                    "action": "isolate",
                    "priority": 2
                },
                {
                    "name": "medium_risk_challenge",
                    "condition": "risk_level == 'medium'",
                    "action": "challenge",
                    "priority": 3
                },
                {
                    "name": "low_risk_allow",
                    "condition": "risk_level == 'low'",
                    "action": "allow",
                    "priority": 4
                }
            ],
            "thresholds": {
                "anomaly_threshold": 0.7,
                "isolation_timeout": 300,
                "challenge_timeout": 60
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                return default_policies
        
        return default_policies
    
    def evaluate_access_request(
        self,
        request: AccessRequest,
        anomaly_score: float,
        historical_data: Optional[Dict] = None
    ) -> PolicyDecision:
        """
        Evaluate an access request and make policy decision.
        
        Args:
            request: Access request to evaluate
            anomaly_score: Anomaly score from HAE model
            historical_data: Historical entity data
            
        Returns:
            Policy decision
        """
        # Check cache first
        cache_key = self._get_cache_key(request, anomaly_score)
        cached_decision = self._get_cached_decision(cache_key)
        if cached_decision:
            return cached_decision
        
        # Calculate risk assessment
        risk_assessment = self.risk_calculator.calculate_risk(
            request.entity,
            anomaly_score,
            historical_data
        )
        
        # Apply policy rules
        action, applied_rules = self._apply_policies(risk_assessment, request)
        
        # Create enforcement details
        enforcement_details = self._create_enforcement_details(
            action, risk_assessment, request
        )
        
        # Create policy decision
        decision = PolicyDecision(
            request_id=request.request_id,
            action=action,
            risk_assessment=risk_assessment,
            policy_rules=applied_rules,
            enforcement_details=enforcement_details,
            timestamp=time.time()
        )
        
        # Cache decision
        self._cache_decision(cache_key, decision)
        
        logger.info(f"Policy decision for {request.entity.entity_id}: {action.value} (risk: {risk_assessment.risk_level.value})")
        
        return decision
    
    def _apply_policies(
        self, 
        risk_assessment: RiskAssessment, 
        request: AccessRequest
    ) -> Tuple[PolicyAction, List[str]]:
        """Apply policy rules based on risk assessment."""
        applied_rules = []
        
        # Sort policies by priority
        policies = sorted(
            self.policies['access_policies'],
            key=lambda p: p.get('priority', 999)
        )
        
        for policy in policies:
            if self._evaluate_condition(policy['condition'], risk_assessment, request):
                action = PolicyAction(policy['action'])
                applied_rules.append(policy['name'])
                return action, applied_rules
        
        # Default action if no policies match
        return PolicyAction.DENY, ['default_deny']
    
    def _evaluate_condition(
        self,
        condition: str,
        risk_assessment: RiskAssessment,
        request: AccessRequest
    ) -> bool:
        """Evaluate policy condition."""
        # Simple condition evaluation
        # In production, use a proper expression evaluator
        
        context = {
            'risk_level': risk_assessment.risk_level.value,
            'risk_score': risk_assessment.risk_score,
            'anomaly_score': risk_assessment.anomaly_score,
            'entity_type': request.entity.entity_type.value,
            'resource': request.resource,
            'action': request.action
        }
        
        try:
            # Replace condition variables
            eval_condition = condition
            for key, value in context.items():
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(key, f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(key, str(value))
            
            # Simple evaluation (in production, use safer expression evaluator)
            return eval(eval_condition)
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _create_enforcement_details(
        self,
        action: PolicyAction,
        risk_assessment: RiskAssessment,
        request: AccessRequest
    ) -> Dict[str, Any]:
        """Create enforcement details based on action."""
        details = {
            'action': action.value,
            'risk_score': risk_assessment.risk_score,
            'risk_factors': risk_assessment.factors,
            'timestamp': time.time()
        }
        
        if action == PolicyAction.ISOLATE:
            details.update({
                'isolation_duration': self.policies['thresholds']['isolation_timeout'],
                'network_policy': 'restrict_all_except_monitoring',
                'alert_soc': True
            })
        elif action == PolicyAction.CHALLENGE:
            details.update({
                'challenge_type': 'mfa',
                'challenge_timeout': self.policies['thresholds']['challenge_timeout'],
                'step_up_required': True
            })
        elif action == PolicyAction.STEP_UP_AUTH:
            details.update({
                'auth_methods': ['biometric', 'hardware_token'],
                'session_timeout': 900  # 15 minutes
            })
        
        return details
    
    def _get_cache_key(self, request: AccessRequest, anomaly_score: float) -> str:
        """Generate cache key for decision."""
        key_data = f"{request.entity.entity_id}:{request.resource}:{request.action}:{anomaly_score:.2f}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_decision(self, cache_key: str) -> Optional[PolicyDecision]:
        """Get cached decision if still valid."""
        if cache_key in self.decision_cache:
            decision, cached_at = self.decision_cache[cache_key]
            if time.time() - cached_at < self.cache_ttl:
                return decision
            else:
                del self.decision_cache[cache_key]
        return None
    
    def _cache_decision(self, cache_key: str, decision: PolicyDecision) -> None:
        """Cache policy decision."""
        self.decision_cache[cache_key] = (decision, time.time())


class ZTAEnforcement:
    """Enforcement point for Zero Trust policies."""
    
    def __init__(self):
        """Initialize enforcement engine."""
        self.active_isolations = {}
        self.active_challenges = {}
        self.enforcement_log = deque(maxlen=1000)
        
    def enforce_decision(self, decision: PolicyDecision) -> Dict[str, Any]:
        """
        Enforce a policy decision.
        
        Args:
            decision: Policy decision to enforce
            
        Returns:
            Enforcement result
        """
        enforcement_result = {
            'request_id': decision.request_id,
            'action': decision.action.value,
            'enforced_at': time.time(),
            'success': False,
            'details': {}
        }
        
        try:
            if decision.action == PolicyAction.ALLOW:
                enforcement_result.update(self._enforce_allow(decision))
            elif decision.action == PolicyAction.DENY:
                enforcement_result.update(self._enforce_deny(decision))
            elif decision.action == PolicyAction.ISOLATE:
                enforcement_result.update(self._enforce_isolate(decision))
            elif decision.action == PolicyAction.CHALLENGE:
                enforcement_result.update(self._enforce_challenge(decision))
            elif decision.action == PolicyAction.STEP_UP_AUTH:
                enforcement_result.update(self._enforce_step_up_auth(decision))
            
            enforcement_result['success'] = True
            
        except Exception as e:
            logger.error(f"Enforcement failed for {decision.request_id}: {e}")
            enforcement_result['error'] = str(e)
        
        # Log enforcement action
        self.enforcement_log.append(enforcement_result)
        
        return enforcement_result
    
    def _enforce_allow(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Enforce allow action."""
        return {
            'message': 'Access granted',
            'session_timeout': 3600,  # 1 hour default
            'monitoring': 'standard'
        }
    
    def _enforce_deny(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Enforce deny action."""
        return {
            'message': 'Access denied',
            'reason': f"Risk level: {decision.risk_assessment.risk_level.value}",
            'retry_after': 300
        }
    
    def _enforce_isolate(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Enforce isolation action."""
        entity_id = decision.risk_assessment.entity_id
        isolation_duration = decision.enforcement_details.get('isolation_duration', 300)
        
        isolation_info = {
            'entity_id': entity_id,
            'isolated_at': time.time(),
            'expires_at': time.time() + isolation_duration,
            'network_policy': 'isolated',
            'decision_id': decision.request_id
        }
        
        self.active_isolations[entity_id] = isolation_info
        
        # In a real implementation, this would trigger:
        # - Network policy changes (firewall rules, VLANs)
        # - Container/pod isolation in Kubernetes
        # - SOC alerts
        
        logger.warning(f"Entity {entity_id} isolated for {isolation_duration} seconds")
        
        return {
            'message': 'Entity isolated',
            'isolation_id': entity_id,
            'duration': isolation_duration,
            'network_restrictions': 'all_except_monitoring'
        }
    
    def _enforce_challenge(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Enforce challenge action."""
        entity_id = decision.risk_assessment.entity_id
        challenge_id = str(uuid.uuid4())
        
        challenge_info = {
            'challenge_id': challenge_id,
            'entity_id': entity_id,
            'challenge_type': 'mfa',
            'created_at': time.time(),
            'expires_at': time.time() + decision.enforcement_details.get('challenge_timeout', 60),
            'attempts': 0,
            'max_attempts': 3
        }
        
        self.active_challenges[challenge_id] = challenge_info
        
        return {
            'message': 'Additional authentication required',
            'challenge_id': challenge_id,
            'challenge_type': 'mfa',
            'timeout': decision.enforcement_details.get('challenge_timeout', 60)
        }
    
    def _enforce_step_up_auth(self, decision: PolicyDecision) -> Dict[str, Any]:
        """Enforce step-up authentication."""
        return {
            'message': 'Step-up authentication required',
            'auth_methods': decision.enforcement_details.get('auth_methods', ['mfa']),
            'session_timeout': decision.enforcement_details.get('session_timeout', 900)
        }
    
    def check_isolation_status(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Check if entity is currently isolated."""
        if entity_id in self.active_isolations:
            isolation = self.active_isolations[entity_id]
            if time.time() < isolation['expires_at']:
                return isolation
            else:
                # Isolation expired
                del self.active_isolations[entity_id]
                logger.info(f"Isolation expired for entity {entity_id}")
        return None
    
    def release_isolation(self, entity_id: str) -> bool:
        """Manually release entity from isolation."""
        if entity_id in self.active_isolations:
            del self.active_isolations[entity_id]
            logger.info(f"Isolation manually released for entity {entity_id}")
            return True
        return False


class ZTAEngine:
    """Main Zero Trust Architecture engine."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize ZTA engine.
        
        Args:
            config_file: Path to configuration file
        """
        self.policy_engine = PolicyEngine(config_file)
        self.enforcement = ZTAEnforcement()
        self.entities = {}
        self.historical_data = defaultdict(dict)
        self.security_events = deque(maxlen=1000)
        
        # Background thread for cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("ZTA Engine initialized")
    
    def register_entity(self, entity: Entity) -> None:
        """Register a new entity in the system."""
        self.entities[entity.entity_id] = entity
        logger.info(f"Registered entity: {entity.entity_id} ({entity.entity_type.value})")
    
    def process_access_request(
        self,
        entity_id: str,
        resource: str,
        action: str,
        anomaly_score: float,
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        """
        Process an access request through the ZTA engine.
        
        Args:
            entity_id: Entity identifier
            resource: Requested resource
            action: Requested action
            anomaly_score: Anomaly score from HAE model
            context: Additional context information
            
        Returns:
            Policy decision
        """
        # Get entity
        entity = self.entities.get(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not registered")
        
        # Update entity last seen
        entity.last_seen = time.time()
        
        # Check if entity is currently isolated
        isolation_status = self.enforcement.check_isolation_status(entity_id)
        if isolation_status:
            return PolicyDecision(
                request_id=str(uuid.uuid4()),
                action=PolicyAction.DENY,
                risk_assessment=RiskAssessment(
                    entity_id=entity_id,
                    risk_level=RiskLevel.HIGH,
                    risk_score=1.0,
                    anomaly_score=1.0,
                    factors={'isolation': 1.0},
                    timestamp=time.time()
                ),
                policy_rules=['entity_isolated'],
                enforcement_details={'reason': 'Entity currently isolated'},
                timestamp=time.time()
            )
        
        # Create access request
        request = AccessRequest(
            request_id=str(uuid.uuid4()),
            entity=entity,
            resource=resource,
            action=action,
            timestamp=time.time(),
            context=context or {}
        )
        
        # Get historical data
        historical_data = self.historical_data.get(entity_id, {})
        
        # Evaluate request
        decision = self.policy_engine.evaluate_access_request(
            request, anomaly_score, historical_data
        )
        
        # Enforce decision
        enforcement_result = self.enforcement.enforce_decision(decision)
        
        # Log security event
        self._log_security_event(request, decision, enforcement_result)
        
        # Update historical data
        self._update_historical_data(entity, anomaly_score)
        
        return decision
    
    def _log_security_event(
        self,
        request: AccessRequest,
        decision: PolicyDecision,
        enforcement_result: Dict[str, Any]
    ) -> None:
        """Log security event."""
        severity = "high" if decision.risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "medium"
        
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type="access_decision",
            entity_id=request.entity.entity_id,
            severity=severity,
            description=f"Access {decision.action.value} for {request.resource}",
            details={
                'request_id': request.request_id,
                'resource': request.resource,
                'action': request.action,
                'risk_score': decision.risk_assessment.risk_score,
                'anomaly_score': decision.risk_assessment.anomaly_score,
                'enforcement_result': enforcement_result
            },
            timestamp=time.time()
        )
        
        self.security_events.append(event)
        
        if severity == "high":
            logger.warning(f"High severity event: {event.description}")
    
    def _update_historical_data(self, entity: Entity, anomaly_score: float) -> None:
        """Update historical data for entity."""
        entity_id = entity.entity_id
        
        if 'access_history' not in self.historical_data[entity_id]:
            self.historical_data[entity_id]['access_history'] = deque(maxlen=100)
        
        if 'typical_locations' not in self.historical_data[entity_id]:
            self.historical_data[entity_id]['typical_locations'] = set()
        
        # Update access history
        self.historical_data[entity_id]['access_history'].append({
            'timestamp': time.time(),
            'anomaly_score': anomaly_score,
            'ip_address': entity.ip_address,
            'location': entity.location
        })
        
        # Update typical locations
        if entity.ip_address:
            self.historical_data[entity_id]['typical_locations'].add(entity.ip_address)
    
    def _cleanup_expired(self) -> None:
        """Background cleanup of expired isolations and challenges."""
        while True:
            try:
                current_time = time.time()
                
                # Clean up expired isolations
                expired_isolations = [
                    entity_id for entity_id, info in self.enforcement.active_isolations.items()
                    if current_time > info['expires_at']
                ]
                
                for entity_id in expired_isolations:
                    del self.enforcement.active_isolations[entity_id]
                    logger.info(f"Cleaned up expired isolation for {entity_id}")
                
                # Clean up expired challenges
                expired_challenges = [
                    challenge_id for challenge_id, info in self.enforcement.active_challenges.items()
                    if current_time > info['expires_at']
                ]
                
                for challenge_id in expired_challenges:
                    del self.enforcement.active_challenges[challenge_id]
                
                time.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
                time.sleep(60)
    
    def get_entity_status(self, entity_id: str) -> Dict[str, Any]:
        """Get current status of an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return {'error': 'Entity not found'}
        
        isolation_status = self.enforcement.check_isolation_status(entity_id)
        historical_data = self.historical_data.get(entity_id, {})
        
        return {
            'entity_id': entity_id,
            'entity_type': entity.entity_type.value,
            'last_seen': entity.last_seen,
            'is_isolated': isolation_status is not None,
            'isolation_info': isolation_status,
            'access_count': len(historical_data.get('access_history', [])),
            'risk_factors': historical_data.get('last_risk_factors', {})
        }
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard information."""
        current_time = time.time()
        
        # Count recent events by severity
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        severity_counts = defaultdict(int)
        for event in recent_events:
            severity_counts[event.severity] += 1
        
        return {
            'total_entities': len(self.entities),
            'active_isolations': len(self.enforcement.active_isolations),
            'active_challenges': len(self.enforcement.active_challenges),
            'recent_events': {
                'total': len(recent_events),
                'by_severity': dict(severity_counts)
            },
            'system_status': 'operational',
            'last_updated': current_time
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize ZTA engine
    zta = ZTAEngine()
    
    # Register test entities
    user_entity = Entity(
        entity_id="user_001",
        entity_type=EntityType.USER,
        identity="alice@company.com",
        device_id="device_123",
        ip_address="192.168.1.100",
        location={'country': 'US', 'city': 'San Francisco'},
        attributes={
            'device': {
                'managed': True,
                'encrypted': True,
                'up_to_date': True
            }
        }
    )
    
    service_entity = Entity(
        entity_id="service_001",
        entity_type=EntityType.SERVICE,
        identity="api-gateway",
        ip_address="10.0.1.50"
    )
    
    zta.register_entity(user_entity)
    zta.register_entity(service_entity)
    
    # Test access requests with different anomaly scores
    print("Testing ZTA Engine...")
    
    # Low risk access
    print("\n1. Low risk access request:")
    decision1 = zta.process_access_request(
        entity_id="user_001",
        resource="/api/documents",
        action="read",
        anomaly_score=0.1
    )
    print(f"Decision: {decision1.action.value} (Risk: {decision1.risk_assessment.risk_level.value})")
    
    # Medium risk access
    print("\n2. Medium risk access request:")
    decision2 = zta.process_access_request(
        entity_id="user_001",
        resource="/api/sensitive-data",
        action="write",
        anomaly_score=0.5
    )
    print(f"Decision: {decision2.action.value} (Risk: {decision2.risk_assessment.risk_level.value})")
    
    # High risk access
    print("\n3. High risk access request:")
    decision3 = zta.process_access_request(
        entity_id="user_001",
        resource="/api/admin",
        action="delete",
        anomaly_score=0.8
    )
    print(f"Decision: {decision3.action.value} (Risk: {decision3.risk_assessment.risk_level.value})")
    
    # Critical risk access
    print("\n4. Critical risk access request:")
    decision4 = zta.process_access_request(
        entity_id="user_001",
        resource="/api/financial-data",
        action="export",
        anomaly_score=0.95
    )
    print(f"Decision: {decision4.action.value} (Risk: {decision4.risk_assessment.risk_level.value})")
    
    # Try to access while isolated
    print("\n5. Access request while isolated:")
    try:
        decision5 = zta.process_access_request(
            entity_id="user_001",
            resource="/api/documents",
            action="read",
            anomaly_score=0.1
        )
        print(f"Decision: {decision5.action.value}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Check entity status
    print("\n6. Entity status:")
    status = zta.get_entity_status("user_001")
    print(f"Entity status: {status}")
    
    # Get security dashboard
    print("\n7. Security dashboard:")
    dashboard = zta.get_security_dashboard()
    print(f"Dashboard: {dashboard}")
    
    print("\nZTA Engine testing completed!")